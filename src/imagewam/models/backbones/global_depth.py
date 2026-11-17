"""Metric geometry readout. No future RGB, depth, or pose is a model input."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .supervision_masks import masked_mean
from .pose_objective import rotation_residual, translation_residual, rotation_angle


def backproject(depth, K):
    b,h,w=depth.shape
    y,x=torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
    pixels=torch.stack((x+.5,y+.5,torch.ones_like(x)),dim=0).float().reshape(1,3,-1).expand(b,-1,-1)
    rays=torch.linalg.solve(K.float(),pixels).reshape(b,3,h,w)
    return rays*depth.float()[:,None]


def rotation_matrix(d6):
    a,b=d6.float().chunk(2,dim=-1)
    x=F.normalize(a,dim=-1,eps=1e-6)
    y=F.normalize(b-(x*b).sum(-1,keepdim=True)*x,dim=-1,eps=1e-6)
    return torch.stack((x,y,torch.cross(x,y,dim=-1)),dim=-1)


def metric_depth_from_logits(logits, inverse_scale):
    """Reciprocal of squareplus(logits), evaluated without cancellation.

    squareplus(x) = (x + sqrt(x*x + 4))/2. Positive large logits retain
    inverse-depth semantics. Zero/negative logits have finite, nonzero
    depth gradients, without ReLU dead zones or a hard depth plateau.
    """
    x=logits.float()
    root=torch.hypot(x,torch.full_like(x,2.))
    canonical_depth=torch.where(x>=0,2./(root+x.clamp_min(0)),(root-x)*.5)
    return canonical_depth/inverse_scale.float()


class GlobalDepthRegisters(nn.Module):
    def __init__(self,cfg,video_dim,action_dim):
        super().__init__()
        self.cfg=dict(cfg); self.grid=tuple(cfg.get('grid',[11,20])); self.resolution=tuple(cfg.get('resolution',[256,448]))
        self.horizon=int(cfg.get('horizon',16)); n=self.grid[0]*self.grid[1]
        self.global_tokens=nn.Parameter(torch.randn(1,n,action_dim)*.02)
        self.pose_tokens=nn.Parameter(torch.randn(1,self.horizon,video_dim)*.02)
        self.rotation_head=nn.Sequential(nn.LayerNorm(video_dim),nn.Linear(video_dim,6))
        nn.init.zeros_(self.rotation_head[-1].weight)
        with torch.no_grad(): self.rotation_head[-1].bias.copy_(torch.tensor([1.,0,0,0,1,0]))
        self.predict_translation=bool(cfg.get('predict_translation',False))
        if self.predict_translation:
            self.translation_head=nn.Sequential(nn.LayerNorm(video_dim),nn.Linear(video_dim,3))
            nn.init.zeros_(self.translation_head[-1].weight)
            nn.init.zeros_(self.translation_head[-1].bias)
        source=Path(cfg['depth_pro_source'])/'src/depth_pro/network/decoder.py'
        spec=importlib.util.spec_from_file_location('imagewam_depth_pro_decoder',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        # Infer exact decoder dimensions from the official pretrained checkpoint.
        weights=torch.load(cfg['depth_pro_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        dims=[256]+[weights[f'decoder.convs.{i}.weight'].shape[1] for i in range(1,5)]
        self.decoder=module.MultiresConvDecoder(dims_encoder=dims,dim_decoder=256)
        self.head=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.ConvTranspose2d(128,128,2,stride=2),
                               nn.Conv2d(128,32,3,padding=1),nn.ReLU(True),nn.Conv2d(32,1,1),nn.Identity())
        for name in ('decoder','head'):
            getattr(self,name).load_state_dict({k[len(name)+1:]:v for k,v in weights.items() if k.startswith(name+'.')},strict=True)
        self.adapters=nn.ModuleList([nn.Sequential(nn.LayerNorm(action_dim),nn.Linear(action_dim,d)) for d in dims])
        self.snapshot_layers=(4,9,14,19,24)

    def decode(self,snapshots,K):
        if len(snapshots)!=5: raise ValueError('All five register snapshots are required')
        gh,gw=self.grid; h,w=self.resolution
        features=[]
        for i,(adapter,z) in enumerate(zip(self.adapters,snapshots)):
            f=adapter(z.to(adapter[0].weight.dtype)).transpose(1,2).reshape(z.shape[0],-1,gh,gw)
            features.append(F.interpolate(f,size=(h//(2**(i+1)),w//(2**(i+1))),mode='bilinear',align_corners=False))
        def readout(*xs): return self.head(self.decoder(list(xs))[0])[:,0]
        canonical=checkpoint(readout,*features,use_reentrant=False) if torch.is_grad_enabled() else readout(*features)
        prediction=metric_depth_from_logits(canonical,(w/K[:,0,0].float())[:,None,None])
        if self.cfg.get('debug_depth_range',False):
            self._debug_inverse_before_clamp=prediction.detach().reciprocal()
        return prediction

    def losses(self,snapshots,pose,sample):
        K=sample['geometry_K'].to(pose.device).float()
        prediction=self.decode(snapshots,K)
        gt=sample['geometry_depth'].to(pose.device).float()
        valid=sample['geometry_valid'].to(pose.device).bool() & torch.isfinite(gt) & (gt>0)
        # Both point maps are in the FUTURE camera coordinate system, metres.
        error=F.smooth_l1_loss(backproject(prediction,K),backproject(torch.where(valid,gt,0),K),reduction='none',beta=self.cfg.get('xyz_beta',.1)).mean(1)
        counts=valid.flatten(1).sum(1)
        per_sample=(error*valid).flatten(1).sum(1)/counts.clamp_min(1)
        xyz=masked_mean(per_sample,counts>0)
        R=rotation_matrix(self.rotation_head(pose.to(self.rotation_head[0].weight.dtype)))
        target=sample['geometry_rotation'].to(pose.device).float()
        # Keep the legacy chordal loss unless the metric pose objective is selected.
        rotation_valid=sample.get('geometry_rotation_valid',torch.ones(pose.shape[:2],device=pose.device,dtype=torch.bool)).to(pose.device).bool()
        # Mask missing poses before computing errors, including non-finite placeholders.
        target=torch.where(rotation_valid[...,None,None],target,torch.eye(3,device=pose.device))
        rotation_errors=rotation_residual(R,target,self.cfg)
        rotation_counts=rotation_valid.sum(1)
        per_sample_rotation=torch.where(rotation_valid,rotation_errors,0.).sum(1)/rotation_counts.clamp_min(1)
        rot=masked_mean(per_sample_rotation,rotation_counts>0)
        translation_metrics={}
        trans=pose.sum()*0.
        if self.predict_translation:
            translation=self.translation_head(pose.to(self.translation_head[0].weight.dtype)).float()
            translation_valid=sample['geometry_translation_valid'].to(pose.device).bool()
            translation_target=sample['geometry_translation'].to(pose.device).float()
            translation_target=torch.where(translation_valid[...,None],translation_target,0.)
            translation_errors=translation_residual(translation,translation_target,self.cfg)
            translation_counts=translation_valid.sum(1)
            per_sample_translation=torch.where(translation_valid,translation_errors,0.).sum(1)/translation_counts.clamp_min(1)
            trans=masked_mean(per_sample_translation,translation_counts>0)
            distance=(translation.detach()-translation_target).norm(dim=-1)
            translation_metrics=dict(loss_relative_translation=float(trans.detach()),
                translation_error_m=float(masked_mean(distance,translation_valid)),
                translation_endpoint_error_m=float(masked_mean(distance[:,-1],translation_valid[:,-1])),
                translation_identity_error_m=float(masked_mean(translation_target.norm(dim=-1),translation_valid)),
                translation_valid_steps=float(translation_counts.sum()))
        if getattr(self,'audit_components',False): self.last_loss_components=(xyz,rot,trans)
        angles=rotation_angle(R.detach(),target)
        identity_angles=rotation_angle(torch.eye(3,device=target.device).expand_as(target),target)
        # One pose objective and weight; retain separate diagnostics in native units.
        pose_loss=(rot+trans)*(.5 if self.cfg.get('pose_loss_type')=='metric_normalized' else 1.)
        pose_weight=self.cfg.get('lambda_pose',self.cfg.get('lambda_rotation',.05))
        total=self.cfg.get('lambda_xyz',.1)*xyz+pose_weight*pose_loss
        metrics={'loss_global_xyz':float(xyz.detach()),'loss_relative_rotation':float(rot.detach()),
                 'loss_relative_pose':float(pose_loss.detach()),
                 'loss_relative_pose_weighted':float((pose_weight*pose_loss).detach()),
                 'rotation_error_deg':float(masked_mean(angles,rotation_valid)*180/torch.pi),
                 'rotation_endpoint_error_deg':float(masked_mean(angles[:,-1],rotation_valid[:,-1])*180/torch.pi),
                 'rotation_identity_error_deg':float(masked_mean(identity_angles,rotation_valid)*180/torch.pi),
                 'rotation_valid_samples':float((rotation_counts>0).sum()),
                 'rotation_valid_steps':float(rotation_counts.sum()),
                 'rotation_endpoint_valid_samples':float(rotation_valid[:,-1].sum()),
                 'depth_abs_error_m':float((prediction.detach()-gt).abs()[valid].sum()/valid.sum().clamp_min(1)),
                 'geometry_valid_samples':float((counts>0).sum()),
                 'loss_geometry':float(total.detach())}
        metrics.update(translation_metrics)
        if self.cfg.get('debug_depth_range',False):
            inv=self._debug_inverse_before_clamp
            floored=valid & (inv<=1e-4)
            regular=valid & ~floored
            absolute=(prediction.detach()-gt).abs()
            denominator=valid.sum().clamp_min(1)
            metrics.update(
                depth_inverse_zero_fraction=float(((inv==0)&valid).sum()/denominator),
                depth_inverse_floor_fraction=float(floored.sum()/denominator),
                depth_floor_abs_error_share=float(absolute[floored].sum()/absolute[valid].sum().clamp_min(1e-12)),
                depth_nonfloor_abs_error_m=float(absolute[regular].sum()/regular.sum().clamp_min(1)),
                depth_prediction_min_m=float(prediction.detach()[valid].min()) if valid.any() else 0.,
                depth_prediction_max_m=float(prediction.detach()[valid].max()) if valid.any() else 0.,
                depth_gt_max_m=float(gt[valid].max()) if valid.any() else 0.)
            del self._debug_inverse_before_clamp
        return total,metrics
