"""FLUX pose queries and ActionDiT global geometry queries, jointly trained."""
import torch
from .imagewam import ImageWAM
from .global_depth import GlobalDepthRegisters,backproject,rotation_matrix


class ImageWAMGlobalDepth(ImageWAM):
    def _pad_native_observation(self,image):
        valid=self.geometry_config.get('valid_resolution')
        if valid is None:return image
        output=tuple(self.geometry_config['resolution'])
        if tuple(valid)==output and tuple(image.shape[-2:])==(360,640):
            is_single=image.ndim==3
            resized=torch.nn.functional.interpolate(image[None] if is_single else image,size=output,mode='bilinear',align_corners=False,antialias=True)
            return resized[0] if is_single else resized
        if tuple(image.shape[-2:])==tuple(valid):
            return torch.nn.functional.pad(image,(0,output[1]-valid[1],0,output[0]-valid[0]),mode='replicate')
        if tuple(image.shape[-2:])!=output:
            raise ValueError(f'Expected native {valid} or padded {output}, got {image.shape[-2:]}')
        return image

    @torch.no_grad()
    def infer_video_flux2(self,prompt,input_image,**kwargs):
        result=super().infer_video_flux2(prompt,self._pad_native_observation(input_image),**kwargs)
        valid=self.geometry_config.get('valid_resolution')
        if valid is not None:result['image']=result['image'][...,:valid[0],:valid[1]]
        return result

    def configure_geometry(self,cfg):
        if self.stack!='flux2': raise ValueError('Only FLUX.2 is supported')
        self.geometry_config=dict(cfg)
        self.geometry_supervision=GlobalDepthRegisters(cfg,self.video_expert.hidden_dim,self.action_expert.hidden_dim).to(self.device)
        if len(self.video_expert.blocks)!=25: raise ValueError('Expected all 25 FLUX.2 layers')

    def apply_trainable_policy(self):
        super().apply_trainable_policy()
        if getattr(self,'geometry_supervision',None) is not None:
            self.geometry_supervision.train().requires_grad_(True)
            if self.geometry_supervision.depth_enabled and self.geometry_config.get('freeze_depth_decoder',False):
                self.geometry_supervision.decoder.requires_grad_(False)
                self.geometry_supervision.head.requires_grad_(False)

    def extra_checkpoint_payload(self):
        return {'geometry_supervision':self.geometry_supervision.state_dict(),
                'geometry_supervision_config':self.geometry_config,
                'geometry_checkpoint_version':self._geometry_checkpoint_version()}

    def _geometry_checkpoint_version(self):
        geo=self.geometry_supervision
        if not (geo.pose_enabled and geo.depth_enabled): return 5
        return 4 if geo.predict_translation else 3

    def load_extra_checkpoint_payload(self,payload):
        if 'geometry_supervision' in payload:
            expected=self._geometry_checkpoint_version()
            if payload.get('geometry_checkpoint_version')!=expected: raise ValueError('Incompatible geometry checkpoint: pose heads or depth parameterization differ')
            saved=payload.get('geometry_supervision_config',{})
            for key in ('pose_enabled','depth_enabled'):
                if bool(saved.get(key,True))!=bool(self.geometry_config.get(key,True)):
                    raise ValueError(f'Checkpoint branch mismatch: {key}')
            self.geometry_supervision.load_state_dict(payload['geometry_supervision'],strict=True)

    def _joint_register_forward(self,video_pre,clean_pre,action_pre,return_cache=False):
        # Same original-token modulation and backbone checkpoint policy as baseline.
        from .aligned_geometry_forward import aligned_forward
        return aligned_forward(self,video_pre,clean_pre,action_pre,return_cache)

    def _make_clean_pre(self,inputs):
        ref=inputs['ref_image_latents']; ids=inputs['ref_img_ids']
        return self.video_expert.pre_dit(x=ref[:,:0],timestep=ref.new_zeros(ref.shape[0]),
            context=inputs['text_hidden_states'],context_mask=inputs['text_attention_mask'],
            ref_image_hidden_states=ref,target_img_ids=ids[:,:0],ref_img_ids=ids)

    def _forward_flux2_video_only(self,video_pre,attention_mask):
        # Keep inherited image sampling on the same rotation-conditioned path.
        from flux2.model import timestep_embedding
        tr=self.video_expert.transformer;txt=video_pre['tokens']['txt'];b=txt.shape[0]
        vec=tr.time_in(timestep_embedding(txt.new_zeros(b),256))
        single,_=tr.single_stream_modulation(vec)
        clean=dict(video_pre)
        clean['tokens']={'txt':txt,'img':video_pre['tokens']['img'][:,:video_pre['cond_len']]}
        clean['freqs']={'txt':video_pre['freqs']['txt'],'img':video_pre['freqs']['img'][:,:,:video_pre['cond_len']]}
        clean['target_len']=0
        clean['t_mod']={'vec':vec,'double_img':tr.double_stream_modulation_img(vec),
                        'double_txt':tr.double_stream_modulation_txt(vec),'single':single}
        empty=self.action_expert.pre_dit(txt.new_zeros(b,0,self.action_expert.action_dim),txt.new_zeros(b))
        return self._joint_register_forward(video_pre,clean,empty)['video']

    def _training_loss_flux2(self,sample,tiled=False):
        inputs=self.build_inputs_flux2(sample,tiled=tiled)
        z=inputs['target_latent']; a=inputs['action']; b=z.shape[0]
        nv=torch.randn_like(z); na=torch.randn_like(a)
        tv=self.train_video_scheduler.sample_training_t(batch_size=b,device=self.device,dtype=z.dtype)
        ta=self.train_action_scheduler.sample_training_t(batch_size=b,device=self.device,dtype=a.dtype)
        vp=self.video_expert.pre_dit(x=self.train_video_scheduler.add_noise(z,nv,tv),
            timestep=self._scheduler_timestep_to_unit(tv,self.train_video_scheduler),
            context=inputs['text_hidden_states'],context_mask=inputs['text_attention_mask'],
            ref_image_hidden_states=inputs['ref_image_latents'],target_img_ids=inputs['target_img_ids'],ref_img_ids=inputs['ref_img_ids'])
        ap=self.action_expert.pre_dit(self.train_action_scheduler.add_noise(a,na,ta),self._scheduler_timestep_to_unit(ta,self.train_action_scheduler))
        out=self._joint_register_forward(vp,self._make_clean_pre(inputs),ap)
        pv=self.video_expert.post_dit(out['video'],vp); pa=self.action_expert.post_dit(out['action'],ap)
        verr=(pv.float()-self.train_video_scheduler.training_target(z,nv,tv).float()).square().mean(-1)
        if 'image_valid' in sample:
            mask=sample['image_valid'].to(pv.device).float()
            token_valid=torch.nn.functional.avg_pool2d(mask[:,None],16,16).flatten(1)
            assert token_valid.shape==verr.shape
            lv=(verr*token_valid).sum(1)/token_valid.sum(1).clamp_min(1)
        else:
            lv=verr.mean(1)
        from .supervision_masks import masked_mean, image_target_mask
        lv=masked_mean(lv*self.train_video_scheduler.training_weight(tv).to(lv),image_target_mask(sample,lv))
        la=self._compute_action_loss_per_sample(pred_action=pa,target_action=self.train_action_scheduler.training_target(a,na,ta),
            action_is_pad=inputs['action_is_pad'],action_dim_is_pad=inputs.get('action_dim_is_pad'))
        la=(la*self.train_action_scheduler.training_weight(ta).to(la)).mean()
        extra,metrics=self.geometry_supervision.losses(out['snapshots'],out['pose'],sample)
        metrics.update(loss_video=float(lv.detach())*self.loss_lambda_video,loss_action=float(la.detach())*self.loss_lambda_action,
                       loss_video_raw=float(lv.detach()),loss_action_raw=float(la.detach()))
        if getattr(self,'audit_components',False):
            self.last_loss_components=(lv,la,*self.geometry_supervision.last_loss_components)
        return self.loss_lambda_video*lv+self.loss_lambda_action*la+extra,metrics

    @torch.no_grad()
    def infer_geometry_flux2(self,prompt,input_image,K=None,proprio=None,context=None,context_mask=None):
        """Legacy separate-grid mode: RGB is pre-resized and K matches geometry output.

        With valid_resolution configured for joint RGB/geometry resizing, K instead
        matches the supplied input image and is rescaled together with that image.
        """
        self.eval()
        original_hw=tuple(input_image.shape[-2:])
        input_image=self._pad_native_observation(input_image)
        if K is not None and tuple(self.geometry_config.get('valid_resolution',()))==tuple(self.geometry_config['resolution']) and original_hw!=tuple(input_image.shape[-2:]):
            K=K.clone()
            K[...,0,:]*=input_image.shape[-1]/original_hw[1]
            K[...,1,:]*=input_image.shape[-2]/original_hw[0]
        if input_image.ndim==3: input_image=input_image.unsqueeze(0)
        text,mask=self._prepare_flux2_infer_text(prompt,context,context_mask)
        text,mask=self._append_proprio_to_context_if_enabled(context=text,context_mask=mask,proprio=proprio,source='global geometry inference')
        ref,ids=self._encode_flux2_image_tokens(input_image.to(self.device,self.torch_dtype),time_value=10.)
        clean=self._make_clean_pre(dict(ref_image_latents=ref,ref_img_ids=ids,text_hidden_states=text,text_attention_mask=mask))
        empty=self.action_expert.pre_dit(ref.new_zeros(ref.shape[0],0,self.action_expert.action_dim),ref.new_zeros(ref.shape[0]))
        encoded=self._joint_register_forward(clean,clean,empty)
        result={}
        if self.geometry_supervision.depth_enabled:
            if K is None: raise ValueError('Intrinsics K are required for metric depth/XYZ readout')
            K=K.to(self.device).float()
            if K.ndim==2: K=K.unsqueeze(0)
            depth=self.geometry_supervision.decode(encoded['snapshots'],K)
            vh,vw=self.geometry_config.get('valid_resolution',depth.shape[-2:])
            depth=depth[:,:vh,:vw]
            result.update(depth_m=depth.cpu(),xyz_future_camera_m=backproject(depth,K).cpu())
        if self.geometry_supervision.pose_enabled:
            head=self.geometry_supervision.rotation_head
            R=rotation_matrix(head(encoded['pose'].to(head[0].weight.dtype)))
            result['relative_rotation']=R.cpu()
        if self.geometry_supervision.predict_translation:
            head=self.geometry_supervision.translation_head
            result['relative_translation_current_camera_ros_m']=head(encoded['pose'].to(head[0].weight.dtype)).float().cpu()
        return result

    @torch.no_grad()
    def infer_action_flux2(self,prompt,input_image,action_horizon,proprio=None,context=None,context_mask=None,
                          num_inference_steps=20,sigma_shift=None,seed=None,rand_device='cpu'):
        self.eval()
        input_image=self._pad_native_observation(input_image)
        if input_image.ndim==3: input_image=input_image.unsqueeze(0)
        if input_image.shape[0]!=1: raise ValueError('Single observation inference required')
        text,mask=self._prepare_flux2_infer_text(prompt,context,context_mask)
        text,mask=self._append_proprio_to_context_if_enabled(context=text,context_mask=mask,proprio=proprio,source='global-depth inference')
        ref,ids=self._encode_flux2_image_tokens(input_image.to(self.device,self.torch_dtype),time_value=10.)
        clean=self._make_clean_pre(dict(ref_image_latents=ref,ref_img_ids=ids,text_hidden_states=text,text_attention_mask=mask))
        empty=self.action_expert.pre_dit(ref.new_zeros(1,0,self.action_expert.action_dim),ref.new_zeros(1))
        encoded=self._joint_register_forward(clean,clean,empty,return_cache=True)
        generator=None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        a=torch.randn((1,action_horizon,self.action_expert.action_dim),generator=generator,device=rand_device).to(self.device,self.torch_dtype)
        times,deltas=self.infer_action_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps,device=self.device,dtype=a.dtype,shift_override=sigma_shift)
        for t,delta in zip(times,deltas):
            ap=self.action_expert.pre_dit(a,self._scheduler_timestep_to_unit(t.expand(1),self.infer_action_scheduler))
            hidden=ap['tokens']; pe=self.video_expert.transformer.pe_embedder(ap['ids'].to(hidden))
            valid=torch.cat((encoded['cache_valid'],torch.ones(1,action_horizon,device=a.device,dtype=torch.bool)),1)
            attn=valid[:,None,:].expand(-1,action_horizon,-1)
            for idx,(block,cache) in enumerate(zip(self.action_expert.blocks,encoded['cache'])):
                mod=ap['t_mod']['double_img' if idx<self.action_expert.double_layers else 'single']
                state=block.prepare_qkv(hidden,pe,mod)
                mixed=self.mot._mixed_attention(state['q'],torch.cat((cache['k'],state['k']),1),torch.cat((cache['v'],state['v']),1),attn)
                hidden=block.apply_post(mixed,state)
            a=self.infer_action_scheduler.step(self.action_expert.post_dit(hidden,ap),delta,a)
        return {'action':a[0].float().cpu()}
