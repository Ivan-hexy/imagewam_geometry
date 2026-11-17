"""Frozen native VGGT-Omega with differentiable virtual future frame slots."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .geometry_query import VirtualFutureConditioner


def native_depth_only(head, features, images, patch_token_start):
    """Exact native DenseHead depth path; never invoke proj_conf.

    Source: vggt_omega/models/heads/dense_head.py at
    48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6. The confidence path is independent.
    """
    b,s,_,h,w = images.shape
    multi_scale = []
    for fi,li in enumerate(head.intermediate_layer_idx):
        x = features[li][:,:,patch_token_start:].float()
        x = x.reshape(b*s,-1,x.shape[-1])
        x = head.norm(x).permute(0,2,1).reshape(b*s,-1,h//head.patch_size,w//head.patch_size)
        x = head.projects[fi](x)
        x = head._apply_pos_embed(x,w,h)
        multi_scale.append(head.resize_layers[fi](x))
    fused = head._apply_pos_embed(head.scratch_forward(multi_scale),w,h)
    logits = F.pixel_shuffle(head.proj(fused),head.final_shuffle_factor)
    return logits.permute(0,2,3,1).exp().reshape(b,s,h,w)


def relative_transform(E):
    """E is world-to-camera [B,2,3,4]; use native proper rotations."""
    R0,t0 = E[:,0,:,:3],E[:,0,:,3]
    R1,t1 = E[:,1,:,:3],E[:,1,:,3]
    R = R1 @ R0.transpose(-1,-2)
    t = t1 - (R @ t0[...,None]).squeeze(-1)
    top = torch.cat([R,t[...,None]],dim=-1)
    bottom = top.new_zeros(top.shape[0],1,4);bottom[:,:,3] = 1
    return torch.cat([top,bottom],dim=-2)


class ConditionedVGGTOmega(nn.Module):
    def __init__(self, source_path, checkpoint_path, edit_dim=512,
                 activation_checkpointing=True, initial_gate=.01):
        super().__init__()
        if source_path and str(source_path) not in sys.path:
            sys.path.insert(0,str(source_path))
        from vggt_omega.models import VGGTOmega
        with torch.device('meta'):
            omega = VGGTOmega(enable_alignment=False,use_checkpoint=False)
        state = torch.load(Path(checkpoint_path),map_location='cpu',weights_only=True,mmap=True)
        omega.load_state_dict(state,strict=True,assign=True)
        self.omega = omega.eval().requires_grad_(False)
        self.conditioner = VirtualFutureConditioner(edit_dim=edit_dim,initial_gate=initial_gate)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.checkpoint_path = str(checkpoint_path)
        self.source_path = str(source_path)

    def _apply(self,fn,recurse=True):
        # DeepSpeed converts the surrounding trainable model to BF16 globally.
        # Omega's native CameraHead/DenseHead explicitly run FP32 without autocast;
        # retain the original FP32 frozen readout instead of quantizing it and
        # then upcasting. The tiny trainable conditioner follows the caller dtype.
        self.conditioner._apply(fn)
        probe=fn(next(self.omega.parameters()).new_empty(0))
        self.omega.to(device=probe.device)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.omega.eval()  # Frozen pretrained dropout/norm modes never change.
        return self

    def aggregate_slots(self, patch_slots, height, width):
        """Reuse Omega's native alternating frame/global/register blocks verbatim."""
        from vggt_omega.models.aggregator import slice_expand_and_flatten
        agg = self.omega.aggregator
        b,s,n,d = patch_slots.shape
        camera = slice_expand_and_flatten(agg.camera_token,b,s)
        registers = slice_expand_and_flatten(agg.register_token,b,s)
        tokens = torch.cat([camera,registers,patch_slots.reshape(b*s,n,d)],dim=1)
        nt = tokens.shape[1]
        # Positional grids are constants. No trainable-feature operation is under no_grad.
        sin,cos = agg.rope_embed(H=height//agg.patch_size,W=width//agg.patch_size)
        rope = (sin.to(tokens.device,torch.float32),cos.to(tokens.device,torch.float32))
        outputs = []
        for idx in range(agg.depth):
            def block_fn(x, layer_idx=idx):
                x,frame = agg._run_frame_block(x,b,s,nt,d,layer_idx,rope)
                x = agg._run_inter_frame_attention_block(x,b,s,nt,d,layer_idx,agg.inter_frame_attention_types[layer_idx])
                return x,frame
            if self.activation_checkpointing and torch.is_grad_enabled() and tokens.requires_grad:
                tokens,frame = checkpoint(block_fn,tokens,use_reentrant=False)
            else:
                tokens,frame = block_fn(tokens)
            outputs.append(torch.cat([frame,tokens],dim=-1) if idx in agg.cached_layer_indices else None)
        return outputs,agg.patch_token_start

    def forward(self, current_rgb, editing_latent):
        """Only current RGB [B,3,H,W] is accepted. There is no future-image argument."""
        if current_rgb.ndim != 4 or current_rgb.shape[1] != 3:
            raise ValueError('Current geometry RGB must be [B,3,H,W] in [0,1]')
        agg = self.omega.aggregator
        b,_,h,w = current_rgb.shape
        if h%16 or w%16:
            raise ValueError('VGGT-Omega image dimensions must be multiples of 16')
        device_type = current_rgb.device.type
        with torch.autocast(device_type,enabled=(device_type=='cuda'),dtype=torch.bfloat16):
            normalized = (current_rgb[:,None]-agg._resnet_mean)/agg._resnet_std
            patches = agg.patch_embed(normalized[:,0])
            if isinstance(patches,dict):patches = patches['x_norm_patchtokens']
            future = self.conditioner(patches,editing_latent)
            slots = torch.stack([patches,future],dim=1)
            features,start = self.aggregate_slots(slots,h,w)
        with torch.autocast(device_type,enabled=False):
            from vggt_omega.utils.pose_enc import encoding_to_camera
            pose = self.omega.camera_head(features,patch_token_start=start)
            E,K = encoding_to_camera(pose,(h,w))
            T = relative_transform(E)
            future_features = [None if x is None else x[:,1:2] for x in features]
            # DenseHead reads only shape from images; current RGB supplies that shape.
            depth = native_depth_only(self.omega.dense_head,future_features,current_rgb[:,None],start)[:,0]
        return {'T_future_from_current':T,'depth_future':depth,'pose_enc':pose,
                'E_world_to_camera':E,'intrinsics':K,
                'patch_grid':(h//16,w//16),'patch_token_start':start}
