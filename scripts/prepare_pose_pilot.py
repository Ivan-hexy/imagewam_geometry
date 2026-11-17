"""Fit fixed pose scales and create a reproducible 200-update pilot configuration."""
import hashlib, importlib.util, json, os
from pathlib import Path
import h5py, numpy as np

repo=Path(__file__).resolve().parents[1]
src=repo/'src/imagewam'
root=repo/'.pose_pilot'
root.mkdir(exist_ok=True)
data=Path(os.environ.get('SIMPLE_RGBD_ROOT',str(repo/'data/SIMPLE-RGBD-Teleop')))
raw=(data/'meta/episodes.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest() in {'101608685679811fc4c97fa05b14a34578fd886887e4773dea6b3a13c8d2f9a1','97928caf2f45fd9e5343f539459767b08c7138ea94d3657f0a984adcfa0624ba'}, 'Unrecognized episode manifest: audit this dataset before fitting scales'
rows=[json.loads(l) for l in raw.splitlines()]
groups={}
for r in rows:groups.setdefault(r['task_name'],[]).append(r)
heldout=[];train_probe=[];calibration=[]
for task,rs in sorted(groups.items()):
    rs=sorted(rs,key=lambda r:hashlib.sha256(str(r['source_id']).encode()).hexdigest())
    # Entire source trajectory held out, not adjacent windows from that trajectory.
    val_source=rs[0]['source_id']
    heldout.extend(r['episode_index'] for r in rs if r['source_id']==val_source)
    tr=[r for r in rs if r['source_id']!=val_source]
    train_probe.append(tr[0]['episode_index'])
    calibration.extend(tr[i] for i in np.linspace(0,len(tr)-1,8,dtype=int))
train_ids=[r['episode_index'] for r in rows if r['episode_index'] not in heldout]
manifest={'train_episodes':train_ids,'heldout_episodes':heldout,'train_probe_episodes':train_probe,'source_episodes_sha256':hashlib.sha256(raw).hexdigest()}
(root/'split.json').write_text(json.dumps(manifest,indent=2))
spec=importlib.util.spec_from_file_location('dataset',src/'datasets/simple_rgbd.py')
dsmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(dsmod)
angles=[];distances=[]
for r in calibration:
    with h5py.File(data/r['depth_path'],'r') as f:
        w=dsmod.action_windows(*[f[k][:].astype(bool) for k in ['valid_training_frame','valid_observation','valid_adjacent_transition','valid_action','is_synthetic_tail']],32)
        w=w[w[:,2]>0]
        T=f['T_camera_to_world_ros'][:,0]
        for wi in np.linspace(0,len(w)-1,8,dtype=int):
            start,_,cnt,_=map(int,w[wi]);future=T[start+1:start+cnt+1]
            R=T[start,:3,:3].T@future[:,:3,:3]
            theta=np.arccos(np.clip((np.trace(R,axis1=-2,axis2=-1)-1)/2,-1,1))
            t=(T[start,:3,:3].T@(future[:,:3,3]-T[start,:3,3]).T).T
            angles.append(float(theta.mean()));distances.append(float(np.linalg.norm(t,axis=-1).mean()))
sr=float(np.median(angles));st=float(np.median(distances))
assert sr>1e-6 and st>1e-6
cal={'rotation_scale_rad':sr,'translation_scale_m':st,'rotation_scale_deg':sr*180/np.pi,'method':'Median of per-window mean zero-motion errors; 8 training episodes per task, 8 valid windows each. Fixed throughout optimization; no held-out labels used.','calibration_windows':len(angles),'calibration_episodes':[r['episode_index'] for r in calibration]}
(root/'calibration.json').write_text(json.dumps(cal,indent=2))
cfg=f'''# @package _global_
defaults:
  - imagewam_geometry_pose
  - _self_
batch_size: 1
gradient_accumulation_steps: 16
num_workers: 8
max_steps: 200
warmup_steps: 10
learning_rate: 2.5e-5
save_every: 0
eval_every: 0
log_every: 10
checkpoint_mode: none
record_metrics_every_step: true
fixed_loss_eval_steps: [0, 100, 200]
fixed_eval_kind: pose
rank_timer_every: 1
rank_timer_sync_cuda: true
data:
  train:
    tuning_manifest: {root}/split.json
    tuning_partition: train
model:
  geometry_supervision:
    predict_translation: true
    pose_loss_type: metric_normalized
    pose_rotation_scale_rad: {sr}
    pose_translation_scale_m: {st}
    lambda_pose: 0.1
wandb:
  name: github_pose_geodesic_single_200
'''
(repo/'configs/task/pose_geodesic200.yaml').write_text(cfg)

print(json.dumps({'train_episodes':len(train_ids),'heldout_episodes':heldout,'calibration':cal,'config':str(repo/'configs/task/pose_geodesic200.yaml')},indent=2))
