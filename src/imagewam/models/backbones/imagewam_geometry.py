"""Optional training-only geometry supervision; native action inference is inherited."""
from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .imagewam import ImageWAM
from .geometry_query import EditingQueryFusion
from .conditioned_vggt import ConditionedVGGTOmega
from .geometry_losses import geometry_losses


class GeometrySupervision(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.query_fusion = EditingQueryFusion(input_dim=cfg.get('kv_dim',3072),
            hidden_dim=cfg.get('edit_dim',512),num_queries=cfg.get('num_queries',64),
            num_heads=cfg.get('query_heads',8),num_layers=25,
            refine_layers=cfg.get('query_refine_layers',2))
        self.predictor = ConditionedVGGTOmega(cfg['vggt_source'],cfg['vggt_checkpoint'],
            edit_dim=cfg.get('edit_dim',512),activation_checkpointing=cfg.get('activation_checkpointing',True),
            initial_gate=cfg.get('initial_conditioning_gate',.01))

    def apply_trainable_policy(self):
        self.train()
        self.query_fusion.requires_grad_(True)
        self.predictor.conditioner.requires_grad_(True)
        self.predictor.omega.eval().requires_grad_(False)

    def trainable_state_dict(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('predictor.omega.')}


class ImageWAMGeometry(ImageWAM):
    def configure_geometry(self,cfg):
        if self.stack!='flux2':raise ValueError('Geometry supervision currently supports FLUX.2 only')
        self.geometry_config = dict(cfg)
        self.geometry_supervision = GeometrySupervision(self.geometry_config).to(device=self.device)

    def action_visible_editing_cache(self,sample):
        """Exactly infer_action_flux2's t=0, current-only prefill, with autograd.

        This method never reads target_latent, next_frame, actions or geometry labels.
        Text padding is excluded just as it is for the native action attention.
        """
        current = sample.get('current_frame',sample.get('input_image'))
        if current is None:
            current = sample['video'][:,:,0]
        if current.ndim==3:current=current.unsqueeze(0)
        ref,ref_ids = self._encode_flux2_image_tokens(current,time_value=10.)
        context,mask = self._encode_flux2_text(sample)
        proprio = sample.get('proprio')
        if proprio is not None and proprio.ndim==3:proprio=proprio[:,0]
        if self.proprio_encoder is not None:
            context,mask = self._append_proprio_to_context_if_enabled(
                context=context,context_mask=mask,proprio=proprio,source='geometry current-only prefill')
        b = ref.shape[0]
        pre = self.video_expert.pre_dit(x=ref.new_zeros(b,0,ref.shape[-1]),
            timestep=ref.new_zeros(b),context=context,context_mask=mask,
            ref_image_hidden_states=ref,target_img_ids=ref_ids.new_zeros(b,0,ref_ids.shape[-1]),ref_img_ids=ref_ids)
        attention = self._build_mot_attention_mask_flux2(batch_size=b,txt_len=pre['txt_len'],
            target_len=0,cond_len=pre['cond_len'],action_len=0,device=ref.device,text_attention_mask=pre['text_mask'])
        def prefill(txt,img):
            return self.mot.prefill_flux2_video_cache(video_tokens={'txt':txt,'img':img},video_freqs=pre['freqs'],
                video_t_mod=pre['t_mod'],attention_mask=attention)
        if torch.is_grad_enabled() and self.geometry_config.get('checkpoint_prefill',True):
            cache=checkpoint(prefill,pre['tokens']['txt'],pre['tokens']['img'],use_reentrant=False)
        else:
            cache=prefill(pre['tokens']['txt'],pre['tokens']['img'])
        valid = torch.cat([pre['text_mask'],torch.ones(b,pre['cond_len'],device=ref.device,dtype=torch.bool)],dim=1)
        return cache,valid

    def predict_geometry(self,sample,return_cache=False):
        if self.geometry_supervision is None:raise RuntimeError('Geometry module has been removed')
        cache,valid = self.action_visible_editing_cache(sample)
        z = self.geometry_supervision.query_fusion(cache,valid)
        current = sample['geometry_current'].to(device=self.device,dtype=torch.float32,non_blocking=True)
        prediction = self.geometry_supervision.predictor(current,z)
        if return_cache:return prediction,z,cache,valid
        return prediction

    def geometry_training_losses(self,sample):
        cfg=self.geometry_config
        prediction=self.predict_geometry(sample)
        losses=geometry_losses(prediction,sample['geometry_transform'],sample['geometry_depth'],
            rotation_weight=cfg.get('rotation_weight',1.),translation_weight=cfg.get('translation_weight',1.),
            depth_beta=cfg.get('depth_beta',.1))
        total=cfg.get('lambda_pose',1.)*losses['pose']+cfg.get('lambda_depth',.1)*losses['depth']
        metrics={'loss_geometry_pose':float(losses['pose'].detach()),
                 'loss_geometry_depth':float(losses['depth'].detach()),
                 'loss_geometry_rotation':float(losses['rotation'].detach()),
                 'loss_geometry_translation':float(losses['translation'].detach()),
                 'loss_geometry':float(total.detach())}
        return total,metrics,losses,prediction

    def ordinary_training_loss(self,sample,tiled=False):
        if torch.is_grad_enabled() and self.geometry_config.get('checkpoint_imagewam_loss',True):
            # An explicit CUDA tensor lets PyTorch preserve the correct device RNG
            # for the native flow-noise/timestep sampling during recomputation.
            def native_loss(_rng_anchor):
                return ImageWAM.training_loss(self,sample,tiled=tiled)
            anchor=self.proprio_encoder.weight
            return checkpoint(native_loss,anchor,use_reentrant=False,preserve_rng_state=True)
        return ImageWAM.training_loss(self,sample,tiled=tiled)

    def training_loss(self,sample,tiled=False):
        ordinary,metrics=self.ordinary_training_loss(sample,tiled=tiled)
        if self.geometry_supervision is None:return ordinary,metrics
        geometry,extra,_,_=self.geometry_training_losses(sample)
        return ordinary+geometry,{**metrics,**extra}

    def apply_trainable_policy(self):
        super().apply_trainable_policy()
        if getattr(self,'geometry_supervision',None) is not None:
            self.geometry_supervision.apply_trainable_policy()

    def extra_checkpoint_payload(self):
        if self.geometry_supervision is None:return {}
        return {'geometry_supervision':self.geometry_supervision.trainable_state_dict(),
                'geometry_supervision_config':self.geometry_config,
                'geometry_checkpoint_version':1}

    def load_extra_checkpoint_payload(self,payload):
        if self.geometry_supervision is None or 'geometry_supervision' not in payload:return
        result=self.geometry_supervision.load_state_dict(payload['geometry_supervision'],strict=False)
        missing=[x for x in result.missing_keys if not x.startswith('predictor.omega.')]
        if missing or result.unexpected_keys:
            raise RuntimeError(f'Geometry checkpoint mismatch: missing={missing} unexpected={result.unexpected_keys}')

    def remove_geometry_supervision(self):
        """Free training-only weights. Inherited action inference has no branch reference."""
        self.geometry_supervision=None

