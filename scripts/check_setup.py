"""Check the pinned data/cache release, experiment flags and actual tail samples."""
import argparse, ast, hashlib, json, os, sys
from pathlib import Path
from collections import Counter
from hydra import initialize_config_dir, compose
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from imagewam.experiments import EXPERIMENTS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=EXPERIMENTS,default='both')
    parser.add_argument('--config-only',action='store_true')
    parser.add_argument('--require-weights',action='store_true')
    parser.add_argument('--all-windows',action='store_true',help='Enumerate all windows, not just sampled episodes')
    args=parser.parse_args()
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=imagewam_'+args.mode])
    pose,depth=EXPERIMENTS[args.mode];geo=cfg.model.geometry_supervision
    assert cfg.data.train.horizon==geo.horizon==32
    assert bool(geo.enabled)==(pose or depth)
    if geo.enabled:assert bool(geo.pose_enabled)==pose and bool(geo.depth_enabled)==depth
    assert not geo.predict_translation
    assert list(geo.grid)==[11,20]
    assert cfg.model.loss.lambda_video==.5 and cfg.model.loss.lambda_action==1
    if pose:assert geo.lambda_rotation==5
    if depth:assert geo.lambda_xyz==.5
    for p in (ROOT/'src').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    report=dict(mode=args.mode,chunk=32,pose_enabled=pose,depth_enabled=depth,pose_tokens=32 if pose else 0,geometry_tokens=220 if depth else 0,translation_enabled=False,config_passed=True)
    if args.require_weights:
        needed=[cfg.model.flux2_model_path,cfg.model.ae_model_path,cfg.model.action_dit_pretrained_path]
        if depth:needed.append(geo.depth_pro_checkpoint)
        for p in needed:
            if not Path(p).is_file():raise FileNotFoundError(p)
    if not args.config_only:
        release=json.loads((ROOT/'configs/dataset_release.json').read_text())
        data=Path(cfg.data.train.root);cache=Path(cfg.data.train.text_cache_dir).parent
        for relative,expected in release['files'].items():
            dirname,rest=relative.split('/',1);path=(data if dirname=='SIMPLE-RGBD-Teleop' else cache)/rest
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual!=expected:raise ValueError(f'Dataset/cache version mismatch: {path}; expected {expected}, got {actual}. Use a new download directory, not mixed old/new files.')
        episodes=[json.loads(x) for x in (data/'meta/episodes.jsonl').read_bytes().splitlines() if x.strip()]
        assert len(episodes)==release['episodes'] and sum(x['length'] for x in episodes)==release['frames']
        assert [x['episode_index'] for x in episodes]==list(range(release['episodes']))
        manifest=json.loads((cache/'all_verified_training_manifest.json').read_text())
        assert manifest['source_episodes_sha256']==release['files']['SIMPLE-RGBD-Teleop/meta/episodes.jsonl']
        assert manifest['episodes']==len(episodes)
        chosen={}
        for ep in episodes:
            chosen.setdefault(ep['task_name'],ep['episode_index'])
            for key in ['video_path','depth_path','data_path','trajectory_path','metadata_path']:
                if not (data/ep[key]).is_file():raise FileNotFoundError(data/ep[key])
        assert len(chosen)==release['tasks']
        from imagewam.datasets.simple_rgbd import SimpleRGBDDataset
        import torch
        torch.set_num_threads(2)
        ds=SimpleRGBDDataset(root=data,text_cache_dir=cache/'qwen3',horizon=32,
            episode_indices=None if args.all_windows else list(chosen.values()),norm_stats_path=cfg.data.train.norm_stats_path)
        checked=[]
        for ei,end in enumerate(ds.offsets):
            if ds.episodes[ei]['episode_index'] not in chosen.values():continue
            offset=0 if ei==0 else ds.offsets[ei-1]
            for index in [offset,end-1]:
                sample=ds[index];start,count,rotation_count,target_valid=map(int,ds.windows[ei][index-offset])
                assert sample['action'].shape==(32,36)
                assert int((~sample['action_is_pad']).sum())==count>0
                assert int(sample['geometry_rotation_valid'].sum())==rotation_count
                assert bool(sample['image_target_valid'])==bool(target_valid)
                if not target_valid:assert not sample['geometry_valid'].any() and sample['image_is_pad'][-1]
            checked.append(ds.episodes[ei]['episode_index'])
        if args.all_windows:assert len(ds)==release['valid_action_windows_h32'],(len(ds),release['valid_action_windows_h32'])
        for h,v in ds._handles.values():h.close();v.close()
        report.update(dataset_revision=release['revision'],episodes=len(episodes),frames=release['frames'],tasks=dict(Counter(r['task_name'] for r in episodes)),sampled_episodes=checked,
            sampled_first_and_last_windows_passed=True,actual_windows=len(ds) if args.all_windows else None,
            verification='Four pinned metadata/cache hashes, all referenced episode files present, decoded first/last windows for one episode per task. All-window count only with --all-windows; not a full content hash audit.')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
