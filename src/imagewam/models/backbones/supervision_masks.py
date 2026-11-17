"""Reductions that retain a zero-gradient graph when all targets are absent."""
import torch


def masked_mean(values, valid):
    valid=valid.to(device=values.device,dtype=torch.bool)
    count=valid.sum().detach().clone()
    world=1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # DDP/ZeRO averages gradients across ranks. Normalize by the global
        # valid count, so a rank containing only short tails cannot dilute loss.
        world=torch.distributed.get_world_size()
        torch.distributed.all_reduce(count)
    return torch.where(valid,values,torch.zeros_like(values)).sum()*world/count.clamp_min(1)


def image_target_mask(sample, values):
    if 'image_target_valid' in sample:
        return sample['image_target_valid'].to(values.device).bool()
    if sample.get('image_is_pad') is not None:
        return ~sample['image_is_pad'].to(values.device).bool()[:,-1]
    return torch.ones_like(values,dtype=torch.bool)
