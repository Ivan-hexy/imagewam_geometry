"""Joint depth/translation scale normalization for non-metric monocular labels."""
import torch
from torch.nn import functional as F


def geometry_losses(prediction, target_transform, target_depth,
                    rotation_weight=1.,translation_weight=1.,depth_beta=.1):
    with torch.autocast(prediction['depth_future'].device.type,enabled=False):
        pd = prediction['depth_future'].float()
        td = target_depth.to(device=pd.device,dtype=torch.float32)
        target = target_transform.to(device=pd.device,dtype=torch.float32)
        pred = prediction['T_future_from_current'].float()
        if pred.shape!=(pd.shape[0],4,4) or target.shape!=pred.shape:
            raise ValueError('Relative camera transforms must both be [B,4,4]')
        if not torch.isfinite(target).all():raise ValueError('Nonfinite teacher camera transform')
        if pd.shape != td.shape:
            raise ValueError(f'Student/teacher depth grids differ: {pd.shape} vs {td.shape}')
        valid = torch.isfinite(td)&(td>0)
        if not valid.flatten(1).any(dim=1).all():
            raise ValueError('Every pair must have positive finite teacher depth')
        if not torch.isfinite(pd).all() or not (pd>0).all() or not torch.isfinite(pred).all():
            raise ValueError('Nonfinite geometry prediction')
        mask = valid.float();count = mask.sum(dim=(-2,-1)).clamp_min(1)
        # One positive scale for each sample, applied to BOTH its depth and translation.
        # Apply the exact same convention independently to student and offline teacher.
        ps = (pd*mask).sum(dim=(-2,-1))/count
        ts = (torch.where(valid,td,torch.zeros_like(td))*mask).sum(dim=(-2,-1))/count
        ps = ps.clamp_min(1e-6);ts = ts.clamp_min(1e-6)
        normalized_pd = pd/ps[:,None,None]
        normalized_td = torch.where(valid,td,torch.ones_like(td))/ts[:,None,None]
        R,gtR = pred[:,:3,:3],target[:,:3,:3]
        rotation = (.5*(R-gtR).square().sum(dim=(-2,-1))).mean()
        translation = F.smooth_l1_loss(pred[:,:3,3]/ps[:,None],target[:,:3,3]/ts[:,None],beta=.1)
        residual = F.smooth_l1_loss(normalized_pd.clamp_min(1e-6).log(),
            normalized_td.clamp_min(1e-6).log(),beta=depth_beta,reduction='none')
        depth = ((residual*mask).sum(dim=(-2,-1))/count).mean()
        pose = rotation_weight*rotation+translation_weight*translation
    return {'pose':pose,'depth':depth,'rotation':rotation,'translation':translation,
            'student_scale':ps,'teacher_scale':ts}
