"""Joint forward with baseline-compatible original-token modulation.

Preserves baseline video/action timestep modulation and batching of projections.
Appends P to video and G to action, with the existing directed attention edges.
Removing P/G reduces to the baseline transformer (validated separately).
Pose/geometry register modulation stays t=0; original tokens use baseline times.
The inherited action inference uses the same directed per-layer cache.
"""
import torch

def concat_mod(parts):
    value=parts[0][0]
    if isinstance(value, (tuple,list)):
        return tuple(concat_mod([(m[i],n) for m,n in parts]) for i in range(len(value)))
    return torch.cat([m.expand(-1,n,-1) for m,n in parts],dim=1)

def aligned_forward(self, video_pre, clean_pre, action_pre, return_cache=False):
    from flux2.model import apply_rope
    ve, ae, geo = self.video_expert, self.action_expert, self.geometry_supervision
    txt = video_pre['tokens']['txt']
    nr = video_pre['cond_len']
    ref, future = video_pre['tokens']['img'][:, :nr], video_pre['tokens']['img'][:, nr:]
    b, nt = txt.shape[:2]
    p = geo.pose_tokens.to(txt.dtype).expand(b, -1, -1)
    g = geo.global_tokens.to(txt.dtype).expand(b, -1, -1)
    np_, ng, nf, na = p.shape[1], g.shape[1], future.shape[1], action_pre['tokens'].shape[1]
    img = torch.cat((ref, p, future), 1)
    action = torch.cat((g, action_pre['tokens']), 1)
    pids = txt.new_zeros(b, np_, 4)
    pids[:, :, 0] = 3
    pids[:, :, 1] = torch.arange(1, np_+1, device=txt.device)
    gids = ve.build_img_ids(b, *geo.grid, time_value=0, device=txt.device, dtype=txt.dtype)
    ppe = ve.transformer.pe_embedder(pids)
    gpe = ve.transformer.pe_embedder(gids[:, :ng])
    tpe = video_pre['freqs']['txt']
    imgpe = torch.cat((video_pre['freqs']['img'][:, :, :nr], ppe, video_pre['freqs']['img'][:, :, nr:]), 2)
    ape = torch.cat((gpe, ve.transformer.pe_embedder(action_pre['ids'].to(txt.dtype))), 2)
    nc = nt+nr
    ps, fs, gs, aas = nc, nc+np_, nc+np_+nf, nc+np_+nf+ng
    total = aas+na
    mask = torch.zeros(b, total, total, device=txt.device, dtype=torch.bool)
    mask[:, :nc, :nc] = True
    mask[:, ps:fs, :fs] = True
    mask[:, fs:gs, :gs] = True
    mask[:, gs:aas, :fs] = True
    mask[:, gs:aas, gs:aas] = True
    mask[:, aas:, :fs] = True
    mask[:, aas:, gs:] = True
    mask[:, :, :nt] &= video_pre['text_mask'][:, None, :]
    snapshots, caches = [], []
    vm, am = video_pre['t_mod'], action_pre['t_mod']
    cm = clean_pre['t_mod']
    gm = ae.pre_dit(action.new_zeros(b,0,ae.action_dim),action.new_zeros(b))['t_mod']
    condition_mod = vm
    video_double=concat_mod([(condition_mod['double_img'],nr),(cm['double_img'],np_),(vm['double_img'],nf)])
    video_single=concat_mod([(condition_mod['single'],nt+nr),(cm['single'],np_),(vm['single'],nf)])
    action_double=concat_mod([(gm['double_img'],ng),(am['double_img'],na)])
    action_single=concat_mod([(gm['single'],ng),(am['single'],na)])
    flat = self.mot._flux2_flatten_heads
    for idx in range(len(ve.blocks)):
        if idx < ve.double_layers:
            vb, ab = ve.double_blocks[idx], ae.double_blocks[idx]
            q, k, v, pe, num_txt, mods = vb._prepare_qkv(img, txt, imgpe, tpe, video_double, condition_mod['double_txt'])
            q, k = apply_rope(q, k, pe)
            vs = dict(q=flat(q), k=flat(k), v=flat(v))
            act = ab.prepare_qkv(action, ape, action_double)
        else:
            j = idx-ve.double_layers
            vb, ab = ve.single_blocks[j], ae.single_blocks[j]
            vs = self.mot._flux2_video_single_io(vb, torch.cat((txt, img), 1), torch.cat((tpe, imgpe), 2), video_single)
            act = ab.prepare_qkv(action, ape, action_single)
        q, k, v = [torch.cat((vs[key], act[key]), 1) for key in ('q', 'k', 'v')]
        mixed = self.mot._mixed_attention(q, k, v, mask)
        vo, ao = mixed[:, :gs], mixed[:, gs:]
        if idx < ve.double_layers:
            img, txt = vb._apply_residuals(img, txt, vo[:, nt:], vo[:, :nt], mods)
        else:
            combined = vb._out(vs['residual_x'], vo, vs['mlp'], vs['gate'])
            txt, img = combined[:, :nt], combined[:, nt:]
        action = ab.apply_post(ao, act)
        if idx in geo.snapshot_layers:
            snapshots.append(action[:, :ng])
        if return_cache:
            caches.append({key: torch.cat((value[:, :fs], value[:, gs:aas]), 1) for key, value in [('k', k), ('v', v)]})
    valid = torch.cat((video_pre['text_mask'], torch.ones(b, nr+np_+ng, device=txt.device, dtype=torch.bool)), 1)
    return dict(video=dict(txt=txt, img=torch.cat((img[:, :nr], img[:, nr+np_:]), 1)),
                action=action[:, ng:], pose=img[:, nr:nr+np_], snapshots=snapshots, cache=caches, cache_valid=valid)
