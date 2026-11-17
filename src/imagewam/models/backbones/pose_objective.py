"""Metric-aligned pose residuals; fixed scales are fitted on training labels only.

References: 6DRepNet (geodesic rotation), Kendall & Cipolla CVPR 2017
(translation/rotation balancing). This fixed-normalization combination is our
adaptation, not a verbatim implementation of either paper.
"""
import math
import torch


def rotation_angle(pred, target):
    # atan2 avoids the acos clamp dead zone at small angles. Norm has the zero
    # subgradient at identity. SO(3) geodesic distance is non-smooth at pi.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        delta = pred.float().transpose(-1, -2) @ target.float()
        skew = torch.stack((delta[..., 2, 1]-delta[..., 1, 2],
                            delta[..., 0, 2]-delta[..., 2, 0],
                            delta[..., 1, 0]-delta[..., 0, 1]), -1)
        sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
        cosine = (delta.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*0.5
        return torch.atan2(sine, cosine)


def checked_scale(cfg, name):
    scale = float(cfg[name])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return scale


def rotation_residual(pred, target, cfg):
    if cfg.get('pose_loss_type', 'legacy') == 'metric_normalized':
        return rotation_angle(pred, target) / checked_scale(cfg, 'pose_rotation_scale_rad')
    return (pred-target).square().sum((-1, -2))/8


def translation_residual(pred, target, cfg):
    if cfg.get('pose_loss_type', 'legacy') == 'metric_normalized':
        return torch.linalg.vector_norm(pred.float()-target.float(), dim=-1) / checked_scale(cfg, 'pose_translation_scale_m')
    if cfg.get('translation_loss_type') == 'scaled_l2':
        scale = float(cfg.get('translation_scale_m', 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('translation_scale_m must be positive')
        return (pred-target).square().sum(-1)/(4*scale**2)
    return torch.nn.functional.smooth_l1_loss(pred,target,reduction='none',beta=cfg.get('translation_beta',.05)).mean(-1)
