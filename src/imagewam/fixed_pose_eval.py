"""Fixed trajectory-held-out probes, separate from training probes."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from omegaconf import OmegaConf
from .datasets.simple_rgbd import SimpleRGBDDataset
from .models.backbones.global_depth import rotation_matrix
from .models.backbones.pose_objective import rotation_angle


def run_fixed_eval(trainer):
    cfg=trainer.cfg;model=trainer.accelerator.unwrap_model(trainer.model)
    assert trainer.accelerator.num_processes==1
    if not hasattr(trainer,'_pose_probe'):
        args=OmegaConf.to_container(cfg.data.train,resolve=True);args.pop('_target_')
        split=json.loads(Path(args.pop('tuning_manifest')).read_text());args.pop('tuning_partition')
        probes={}
        for group,key in [('train','train_probe_episodes'),('heldout','heldout_episodes')]:
            args['episode_indices']=split[key];ds=SimpleRGBDDataset(**args);indices=[]
            for ei,r in enumerate(ds.episodes):
                offset=0 if ei==0 else ds.offsets[ei-1]
                eligible=np.flatnonzero(ds.windows[ei][:,2]>0)
                for fraction in (.35,.8):indices.append(offset+int(eligible[int((len(eligible)-1)*fraction)]))
            probes[group]=(ds,indices)
        trainer._pose_probe=probes
    captured={};model.eval()
    hooks=[model.geometry_supervision.translation_head.register_forward_hook(lambda m,i,o:captured.update(t=o.detach())),
           model.geometry_supervision.rotation_head.register_forward_hook(lambda m,i,o:captured.update(r=o.detach()))]
    records=[]
    try:
        with torch.random.fork_rng(devices=[trainer.accelerator.device]),torch.no_grad():
            for group,(ds,indices) in trainer._pose_probe.items():
                for pos,index in enumerate(indices):
                    torch.manual_seed(2026111800+pos)
                    sample=default_collate([ds[index]])
                    with trainer.accelerator.autocast():loss,metrics=model.training_loss(sample)
                    assert torch.isfinite(loss)
                    R=rotation_matrix(captured['r']).float().cpu();t=captured['t'].float().cpu()
                    gtR=sample['geometry_rotation'];gtt=sample['geometry_translation']
                    valid=sample['geometry_rotation_valid'] & sample['geometry_translation_valid']
                    deg=rotation_angle(R,gtR)*180/torch.pi
                    dist=(t-gtt).norm(dim=-1)
                    zero_t=gtt.norm(dim=-1);zero_r=rotation_angle(torch.eye(3).expand_as(gtR),gtR)*180/torch.pi
                    records.append(dict(group=group,episode=int(sample['episode_index'][0]),frames=sample['frame_indices'][0].tolist(),
                        rotation_deg=deg[valid].tolist(),translation_m=dist[valid].tolist(),
                        zero_rotation_deg=zero_r[valid].tolist(),zero_translation_m=zero_t[valid].tolist(),
                        translation_prediction_m=t[valid].tolist(),translation_target_m=gtt[valid].tolist(),
                        rotation_endpoint_deg=float(deg[0,-1]) if valid[0,-1] else None,
                        translation_endpoint_m=float(dist[0,-1]) if valid[0,-1] else None,
                        metrics={k:float(v) for k,v in metrics.items()}))
    finally:
        for hook in hooks:hook.remove()
        trainer._set_dit_only_train_mode()
    result={'step':trainer.global_step,'evaluation':'Fixed RGB/state inputs and diffusion noise; held-out source trajectories excluded from optimizer and loss-scale calibration; not robot rollout success rate.'}
    for group in ['train','heldout']:
        rows=[r for r in records if r['group']==group];agg={'windows':len(rows)}
        for key in ['rotation_deg','translation_m','zero_rotation_deg','zero_translation_m']:
            agg[key+'_mean']=float(np.mean([np.mean(r[key]) for r in rows]))
            agg[key+'_median_step']=float(np.median([v for r in rows for v in r[key]]))
            agg[key+'_p90_step']=float(np.percentile([v for r in rows for v in r[key]],90))
        for key in ['rotation_endpoint_deg','translation_endpoint_m']:
            vals=[r[key] for r in rows if r[key] is not None];agg[key]=float(np.mean(vals)) if vals else None
        for key in rows[0]['metrics']:
            agg[key]=float(np.mean([r['metrics'][key] for r in rows]))
        result[group]=agg
    with (Path(trainer.output_dir)/'fixed_eval.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    with (Path(trainer.output_dir)/'fixed_pose_records.jsonl').open('a') as f:f.write(json.dumps({'step':trainer.global_step,'rows':records})+'\n')
    print('[fixed_pose_eval] '+json.dumps(result),flush=True)
    return result
