"""Strict pairing of native SIMPLE training samples and offline VGGT labels."""
from __future__ import annotations
import bisect,json
from pathlib import Path
import av,h5py,numpy as np,torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from imagewam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def decode_current_frame(video_path,frame_index,fps=50):
    with av.open(str(video_path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        if frame_index:
            container.seek(int((frame_index/fps)/stream.time_base),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(video=0):
            if frame.time is None:raise ValueError('Expected timestamped SIMPLE video')
            index=int(round(frame.time*fps))
            if index==frame_index:
                if abs(frame.time-frame_index/fps)>.005:raise ValueError('Current video timestamp mismatch')
                return frame.to_ndarray(format='rgb24')
            if index>frame_index:break
    raise ValueError(f'Could not decode exact current frame {frame_index} from {video_path}')


class SimpleGeometryDataset(RobotVideoDataset):
    def __init__(self,annotation_root,geometry_resolution=416,**kwargs):
        if kwargs.get('num_frames')!=17 or not kwargs.get('endpoint_frames_only',False):
            raise ValueError('This implementation requires action chunk16 and endpoint RGB supervision')
        if kwargs.get('global_sample_stride',1)!=1 or kwargs.get('sample_index_stride',1)!=1:
            raise ValueError('Geometry labels require unstrided source frame indices')
        if kwargs.get('video_augmentation') or kwargs.get('condition_frame_augmentation'):
            raise ValueError('Geometric crops/augmentations must be synchronized with labels')
        if kwargs.get('nonidle_filter_path') or kwargs.get('episode_index_filter'):
            raise ValueError('Filtered source indexing is not supported by this strict pair index')
        self.annotation_root=Path(annotation_root)
        self.geometry_resolution=int(geometry_resolution)
        self.geometry_episodes=[];self.valid_offsets=[]
        super().__init__(**kwargs)
        global_offset=0;valid=0
        for root in kwargs['dataset_dirs']:
            root=Path(root);info=json.loads((root/'meta/info.json').read_text())
            if info['fps']!=50:raise ValueError('Expected SIMPLE 50Hz labels')
            episodes=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
            for ep in episodes:
                ei,n=int(ep['episode_index']),int(ep['length'])
                pairs=max(0,n-16)
                if pairs:
                    video=root/info['video_path'].format(episode_index=ei,episode_chunk=ei//info.get('chunks_size',1000),video_key='observation.images.egocentric')
                    label=self.annotation_root/root.name/f'episode_{ei:06d}.h5'
                    if not label.is_file():raise FileNotFoundError(label)
                    self.geometry_episodes.append(dict(task=root.name,episode=ei,frames=n,pairs=pairs,
                        global_start=global_offset,valid_start=valid,video=video,label=label))
                    valid+=pairs;self.valid_offsets.append(valid)
                global_offset+=n
        self.geometry_length=valid
        if global_offset!=len(self.lerobot_dataset):
            raise ValueError('Source metadata and native training dataset indexing disagree')

    def __len__(self):
        return getattr(self,'geometry_length',super().__len__())

    def __getitem__(self,index):
        index=int(index)
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        ep=self.geometry_episodes[bisect.bisect_right(self.valid_offsets,index)]
        t=index-ep['valid_start'];assert t+16<ep['frames']
        # _get raises on failure. Never use the native random-sample retry here,
        # because a replacement RGB/state paired with old labels would corrupt supervision.
        sample=super()._get(ep['global_start']+t)
        if sample['image_is_pad'].any() or sample['action_is_pad'].any():
            raise ValueError('Strict geometry pair unexpectedly contains padding')
        with h5py.File(ep['label'],'r') as f:
            if not f.attrs['complete'] or f.attrs['horizon']!=16 or f.attrs.get('schema_version')!=2:
                raise ValueError(f'Incomplete or incompatible teacher labels: {ep["label"]}')
            np.testing.assert_array_equal(f['frame_indices'][t],[t,t+16])
            depth=f['depth_future'][t].astype(np.float32)
            transform=f['T_future_from_current'][t].astype(np.float32)
        image=Image.fromarray(decode_current_frame(ep['video'],t)).convert('RGB')
        from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio,_balanced_target_shape
        image=_crop_to_supported_aspect_ratio(image)
        w,h=image.size;gh,gw=_balanced_target_shape(h/w,self.geometry_resolution,16)
        if tuple(depth.shape)!=(gh,gw):raise ValueError('Teacher depth and current preprocessing grid differ')
        sample.update(geometry_current=to_tensor(image.resize((gw,gh),Image.Resampling.BICUBIC)),
            geometry_depth=torch.from_numpy(depth),geometry_transform=torch.from_numpy(transform),
            geometry_frame_indices=torch.tensor([t,t+16]),geometry_episode_index=ep['episode'],geometry_task=ep['task'])
        return sample
