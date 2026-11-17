"""Strict current-only geometry inputs and t+16 teacher targets for SONIC-8."""
from __future__ import annotations
import bisect
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .lerobot.robot_video_dataset import RobotVideoDataset
from .humanoidarena_pairs import episode_plan, plan_fingerprint, label_path, decode_frame, preprocess, SCHEMA_VERSION


class HumanoidArenaGeometryDataset(RobotVideoDataset):
    def __init__(self, annotation_root, geometry_resolution=416, validation_only=False, **kwargs):
        if len(kwargs['dataset_dirs']) != 1 or kwargs.get('lerobot_backend') != 'v3':
            raise ValueError('Use the single merged native v3 SONIC-8 dataset')
        if kwargs.get('num_frames') != 17 or not kwargs.get('endpoint_frames_only'):
            raise ValueError('Geometry requires t,t+16 images and action chunk 16')
        for key in ['global_sample_stride', 'sample_index_stride', 'action_video_freq_ratio']:
            if kwargs.get(key,1) != 1:
                raise ValueError(f'Geometry labels require {key}=1')
        for key in ['video_augmentation','condition_frame_augmentation','nonidle_filter_path',
                    'episode_index_filter','skip_padding_as_possible','val_set_proportion']:
            if kwargs.get(key):
                raise ValueError(f'Unsupported reindexing or augmentation: {key}')
        self.source_root = Path(kwargs['dataset_dirs'][0])
        self.annotation_root = Path(annotation_root)
        self.geometry_resolution = int(geometry_resolution)
        manifest = json.loads((self.annotation_root/'manifest.json').read_text())
        episodes = episode_plan(self.source_root)
        if manifest['source_fingerprint'] != plan_fingerprint(episodes):
            raise ValueError('Teacher labels belong to different source data')
        if manifest['schema_version'] != SCHEMA_VERSION or manifest['horizon'] != 16:
            raise ValueError('Incompatible geometry labels')
        if manifest['status'] != 'passed' or bool(manifest['pilot']) != bool(validation_only):
            raise ValueError('Incomplete labels or pilot/full training mismatch')
        super().__init__(**kwargs)
        if sum(ep['frames'] for ep in episodes) != len(self.lerobot_dataset):
            raise ValueError('Native training indices differ from episode metadata')
        self.geometry_episodes, self.valid_offsets = [], []
        valid = 0
        for ep in episodes:
            path = label_path(self.annotation_root, ep)
            if validation_only and not path.is_file():
                continue
            with h5py.File(path,'r') as f:
                assert f.attrs['complete'] and f.attrs['schema_version'] == SCHEMA_VERSION
                assert f.attrs['fingerprint'] == manifest['fingerprint']
                assert f.attrs['source_fingerprint'] == manifest['source_fingerprint']
                assert f.attrs['episode'] == ep['episode'] and f.attrs['horizon'] == 16
                indices = f['frame_indices'][:]
                assert len(indices) and np.all(indices[:,1] == indices[:,0]+16)
                assert np.all(np.diff(indices[:,0]) > 0) and indices[0,0] >= 0 and indices[-1,1] < ep['frames']
                np.testing.assert_array_equal(f['global_frame_indices'][:], indices+ep['global_start'])
                if not validation_only:
                    np.testing.assert_array_equal(indices[:,0],np.arange(ep['pairs']))
                ep = dict(ep,label=path,valid_start=valid,source_indices=indices[:,0])
                self.geometry_episodes.append(ep)
                valid += len(indices)
                self.valid_offsets.append(valid)
        self.geometry_length = valid
        assert valid == manifest['pairs']
        self.annotation_manifest = manifest

    def __len__(self):
        return getattr(self,'geometry_length',super().__len__())

    def __getitem__(self,index):
        index = int(index)
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        ep = self.geometry_episodes[bisect.bisect_right(self.valid_offsets,index)]
        row = index-ep['valid_start']
        t = int(ep['source_indices'][row])
        # _get + disabled padding retries: errors must not substitute another sample.
        sample = super()._get(ep['global_start']+t)
        for key in ['image_is_pad','action_is_pad','proprio_is_pad']:
            if sample[key].any(): raise ValueError(f'Unexpected padded geometry sample: {key}')
        with h5py.File(ep['label'],'r') as f:
            np.testing.assert_array_equal(f['frame_indices'][row],[t,t+16])
            depth = f['depth_future'][row].astype(np.float32)
            transform = f['T_future_from_current'][row].astype(np.float32)
        current = preprocess(decode_frame(self.source_root,ep,t),self.geometry_resolution)
        assert current.shape[1:] == depth.shape
        sample.update(geometry_current=current,geometry_depth=torch.from_numpy(depth),
            geometry_transform=torch.from_numpy(transform),geometry_frame_indices=torch.tensor([t,t+16]),
            geometry_episode_index=ep['episode'],geometry_task=ep['task'])
        return sample
