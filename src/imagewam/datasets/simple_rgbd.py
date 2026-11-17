"""Read immutable replay RGB-D directly; never pair with original teleop RGB."""
import bisect, hashlib, json
from collections import OrderedDict
from pathlib import Path
import av, h5py, numpy as np, torch
from torch.nn import functional as F
from torch.utils.data import Dataset

PROMPT="A video recorded from a robot's point of view executing the following instruction: {task}"

def action_windows(valid, obs, transition, action, synthetic, horizon):
    """Keep real tail actions without inventing observations beyond the episode.

    Interior gaps remain rejected. Only the physical end of the episode may
    shorten a window; terminal observations with no action are not start points.
    """
    n=len(valid)
    real_actions=np.flatnonzero(action & ~synthetic)
    stop=int(real_actions[-1])+1 if len(real_actions) else 0
    windows=[]
    for t in range(stop):
        if not (valid[t] and obs[t] and action[t] and not synthetic[t]): continue
        end=min(t+horizon,stop)
        obs_end=min(t+horizon,n-1)
        if not (action[t:end].all() and obs[t:obs_end+1].all()
                and transition[t:obs_end].all() and not synthetic[t:obs_end+1].any()): continue
        windows.append((t,end-t,obs_end-t,t+horizon<n))
    return np.asarray(windows,dtype=np.int64).reshape(-1,4)


class SimpleRGBDDataset(Dataset):
    def __init__(self,root,text_cache_dir,horizon=16,video_size=(176,320),geometry_resolution=(256,448),
                 split='all',max_episodes=None,norm_stats_path=None,native_resolution=False,episode_indices=None,
                 tuning_manifest=None,tuning_partition='train',**unused):
        if unused: raise ValueError(f'Unexpected dataset options: {list(unused)}')
        self.root=Path(root); self.horizon=int(horizon); self.video_size=tuple(video_size)
        self.geometry_resolution=tuple(geometry_resolution); self.text_cache_dir=Path(text_cache_dir)
        self.native_resolution=bool(native_resolution)
        stats_path=Path(norm_stats_path) if norm_stats_path else self.root/'meta/stats.json'
        self.stats=json.loads(stats_path.read_text())
        if self.horizon<1: raise ValueError('horizon must be positive')
        self.episodes=[]; self.offsets=[]; self.starts=[]; self.windows=[]; self._handles=OrderedDict(); self.text={}
        rows=[json.loads(x) for x in (self.root/'meta/episodes.jsonl').read_text().splitlines()]
        selected=[r for r in rows if r['split']==split] if split!='all' else rows
        if tuning_manifest:
            episode_indices=json.loads(Path(tuning_manifest).read_text())[tuning_partition+'_episodes']
        if episode_indices is not None:
            selected=[r for r in selected if r['episode_index'] in set(episode_indices)]
        if max_episodes: selected=selected[:int(max_episodes)]
        total=0
        for r in selected:
            for k in ('depth_path','video_path'):
                if not (self.root/r[k]).is_file(): raise FileNotFoundError(self.root/r[k])
            with h5py.File(self.root/r['depth_path'],'r') as f:
                valid=f['valid_training_frame'][:].astype(bool)
                obs=f['valid_observation'][:].astype(bool)
                transition=f['valid_adjacent_transition'][:].astype(bool)
                action=f['valid_action'][:].astype(bool)
                synthetic=f['is_synthetic_tail'][:].astype(bool)
                windows=action_windows(valid,obs,transition,action,synthetic,self.horizon)
            if not len(windows): continue
            self.episodes.append(r);self.windows.append(windows);self.starts.append(windows[:,0]);total+=len(windows);self.offsets.append(total)
            prompt=PROMPT.format(task=r['tasks'][0])
            if prompt not in self.text:
                digest=hashlib.sha256(prompt.encode()).hexdigest()
                path=self.text_cache_dir/f'{digest}.qwen3_flux2_len128.pt'
                self.text[prompt]=torch.load(path,map_location='cpu',weights_only=True)
        if not total: raise ValueError('No valid RGB-D action windows')

    def __len__(self): return self.offsets[-1]

    def _open(self,r):
        key=r['episode_index']
        if key not in self._handles:
            container=av.open(str(self.root/r['video_path']))
            container.streams.video[0].codec_context.thread_count=1
            self._handles[key]=(h5py.File(self.root/r['depth_path'],'r'),container)
            if len(self._handles)>4:
                _,old=self._handles.popitem(last=False)
                for x in old: x.close()
        self._handles.move_to_end(key)
        return self._handles[key]

    @staticmethod
    def _rgb(container,t):
        stream=container.streams.video[0]
        container.seek(int((t/50)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        for f in container.decode(stream):
            idx=round(float(f.pts*f.time_base)*50)
            if idx==t: return torch.from_numpy(f.to_ndarray(format='rgb24')).permute(2,0,1).float()/255
            if idx>t: break
        raise ValueError(f'Cannot decode exact frame {t}')

    def normalize(self,x,key):
        lo=torch.tensor(self.stats[key]['q01']);hi=torch.tensor(self.stats[key]['q99'])
        span=hi-lo;constant=span<1e-4;span=torch.where(constant,2.,span)
        scale=2/span;offset=torch.where(constant,-lo,-1-scale*lo)
        return (x*scale+offset).clamp(-5,5)

    def __getitem__(self,index):
        if index<0 or index>=len(self): raise IndexError(index)
        ei=bisect.bisect_right(self.offsets,index); start=0 if ei==0 else self.offsets[ei-1]
        r=self.episodes[ei];t,action_count,rotation_count,target_valid=map(int,self.windows[ei][index-start]);end=t+self.horizon
        h5,video=self._open(r)
        # A current-frame placeholder keeps tensor shapes fixed, with no target loss.
        target_index=end if target_valid else t
        current_image=self._rgb(video,t)
        images=torch.stack((current_image,self._rgb(video,end) if target_valid else current_image))
        raw_depth=torch.from_numpy(h5['depth_m'][target_index,0]);raw_valid=torch.from_numpy(h5['depth_valid'][target_index,0])
        if not target_valid: raw_depth=torch.zeros_like(raw_depth);raw_valid=torch.zeros_like(raw_valid)
        K=torch.from_numpy(h5['K'][target_index,0]).float()
        if self.native_resolution:
            assert images.shape[-2:]==raw_depth.shape==torch.Size([360,640])
            assert self.video_size==self.geometry_resolution==(384,640)
            # Right/bottom-only padding preserves the original pixel coordinates and K.
            images=F.pad(images,(0,0,0,24),mode='replicate')*2-1
            depth=F.pad(raw_depth,(0,0,0,24))
            valid=F.pad(raw_valid,(0,0,0,24),value=False)
            image_valid=F.pad(torch.ones(360,640,dtype=torch.bool),(0,0,0,24),value=False)
        else:
            images=F.interpolate(images,size=self.video_size,mode='bilinear',align_corners=False,antialias=True)*2-1
            depth=F.interpolate(raw_depth[None,None],size=self.geometry_resolution,mode='nearest-exact')[0,0]
            valid=F.interpolate(raw_valid[None,None].float(),size=self.geometry_resolution,mode='nearest-exact')[0,0].bool()
            K[0,:]*=self.geometry_resolution[1]/raw_depth.shape[1]
            K[1,:]*=self.geometry_resolution[0]/raw_depth.shape[0]
            image_valid=torch.ones(self.video_size,dtype=torch.bool)
        current_T=h5['T_camera_to_world_ros'][t,0]
        future_T=h5['T_camera_to_world_ros'][t+1:t+rotation_count+1,0]
        current_R=current_T[:3,:3]
        future_R=future_T[:,:3,:3]
        rotations=np.tile(np.eye(3,dtype=np.float32),(self.horizon,1,1))
        rotations[:rotation_count]=np.einsum('ij,tjk->tik',current_R.T,future_R)
        # Metres in the CURRENT camera ROS frame; invariant to world-frame origin/rotation.
        translations=np.zeros((self.horizon,3),dtype=np.float32)
        translations[:rotation_count]=np.einsum('ij,tj->ti',current_R.T,future_T[:,:3,3]-current_T[:3,3])
        action=torch.zeros(self.horizon,h5['action'].shape[-1])
        action[:action_count]=self.normalize(torch.from_numpy(h5['action'][t:t+action_count]).float(),'action')
        proprio=torch.zeros(self.horizon,h5['state'].shape[-1])
        proprio[:action_count]=self.normalize(torch.from_numpy(h5['state'][t:t+action_count]).float(),'states')
        action_is_pad=torch.arange(self.horizon)>=action_count
        prompt=PROMPT.format(task=r['tasks'][0]);text=self.text[prompt]
        return dict(video=images.permute(1,0,2,3),action=action,proprio=proprio,
            prompt=prompt,instruction=prompt,text_hidden_states=text['text_hidden_states'],text_attention_mask=text['text_attention_mask'].bool(),
            action_is_pad=action_is_pad,image_is_pad=torch.tensor([False,not target_valid]),
            image_target_valid=torch.tensor(bool(target_valid)),
            proprio_is_pad=action_is_pad.clone(),geometry_depth=depth,geometry_valid=valid,
            geometry_K=K,geometry_rotation=torch.from_numpy(rotations).float(),
            geometry_rotation_valid=torch.arange(self.horizon)<rotation_count,
            geometry_translation=torch.from_numpy(translations),
            geometry_translation_valid=torch.arange(self.horizon)<rotation_count,
            image_valid=image_valid,frame_indices=torch.tensor([t,end if target_valid else -1]),episode_index=r['episode_index'])
