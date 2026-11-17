"""Validate portable configuration and sampled SIMPLE RGB-D training windows."""
import argparse
import ast
from pathlib import Path
import hashlib
import json
import os
import sys
from collections import Counter
from hydra import initialize_config_dir, compose

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'src'))
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['baseline','geometry'],default='geometry')
parser.add_argument('--config-only',action='store_true')
parser.add_argument('--require-weights',action='store_true')
args=parser.parse_args()
with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
    cfg=compose(config_name='train',overrides=['task=imagewam_'+args.mode])
assert cfg.data.train.horizon==cfg.model.geometry_supervision.horizon==32
assert bool(cfg.model.geometry_supervision.enabled)==(args.mode=='geometry')
assert not cfg.model.geometry_supervision.predict_translation
assert list(cfg.model.geometry_supervision.grid)==[11,20]
for p in (ROOT/'src').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
report={'mode':args.mode,'chunk':32,'geometry_enabled':bool(cfg.model.geometry_supervision.enabled),'translation_enabled':False,'config_passed':True}
if args.require_weights:
    needed=[cfg.model.flux2_model_path,cfg.model.ae_model_path,cfg.model.action_dit_pretrained_path]
    if args.mode=='geometry':needed.append(cfg.model.geometry_supervision.depth_pro_checkpoint)
    for p in needed:
        if not Path(p).is_file():raise FileNotFoundError(p)
if not args.config_only:
    data=Path(cfg.data.train.root);cache=Path(cfg.data.train.text_cache_dir).parent
    raw=(data/'meta/episodes.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='97928caf2f45fd9e5343f539459767b08c7138ea94d3657f0a984adcfa0624ba','Wrong dataset version'
    episodes=[json.loads(x) for x in raw.splitlines() if x.strip()]
    assert len(episodes)==792 and sum(x['length'] for x in episodes)==313553
    manifest=json.loads((cache/'all_verified_training_manifest.json').read_text())
    assert manifest['source_episodes_sha256']==hashlib.sha256(raw).hexdigest()
    assert manifest['episodes']==792
    for ep in episodes:
        for key in ['video_path','depth_path','data_path','trajectory_path','metadata_path']:
            if not (data/ep[key]).is_file():raise FileNotFoundError(data/ep[key])
    from imagewam.datasets.simple_rgbd import SimpleRGBDDataset
    import torch
    torch.set_num_threads(2)
    chosen={}
    for ep in episodes:chosen.setdefault(ep['task_name'],ep['episode_index'])
    ds=SimpleRGBDDataset(root=data,text_cache_dir=cache/'qwen3',horizon=32,
        episode_indices=list(chosen.values()),norm_stats_path=cfg.data.train.norm_stats_path)
    for i,end in enumerate(ds.offsets):
        first=ds[0 if i==0 else ds.offsets[i-1]];last=ds[end-1]
        assert first['action'].shape==last['action'].shape==(32,36)
        assert first['geometry_rotation'].shape==(32,3,3)
        assert (~last['action_is_pad']).sum()>0
        if not last['image_target_valid']:
            assert not last['geometry_valid'].any() and last['image_is_pad'][-1]
    for h,v in ds._handles.values():h.close();v.close()
    report.update(episodes=792,frames=313553,sampled_tasks=len(chosen),sampled_first_and_last_windows_passed=True,
                  verification='Pinned episode manifest hash, complete episode file presence, and sampled loader checks; not a full content hash audit')
print(json.dumps(report,indent=2))
