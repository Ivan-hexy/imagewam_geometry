"""Shared, exact LeRobot-v3 episode/video indexing for teacher and student.

No model imports here: annotation workers can use this module independently.
"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq

HORIZON = 16
VIDEO_KEY = 'observation.images.front'
SCHEMA_VERSION = 3


def episode_plan(root):
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    if info['fps'] != 50 or info['features']['action']['shape'] != [40]:
        raise ValueError('Expected the SONIC-8 50Hz, 40D refpose release')
    rows = []
    for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
        keys = [k for k in pq.read_schema(path).names if not k.startswith('stats/')]
        rows.extend(pq.read_table(path, columns=keys).to_pylist())
    rows.sort(key=lambda x: x['episode_index'])
    tasks = list(dict.fromkeys(row['tasks'][0] for row in rows))
    episodes, offset = [], 0
    for row in rows:
        n, eid = int(row['length']), int(row['episode_index'])
        assert eid == len(episodes) and row['dataset_from_index'] == offset
        assert row['dataset_to_index'] == offset + n and len(row['tasks']) == 1
        prefix = 'videos/' + VIDEO_KEY
        start = float(row[prefix + '/from_timestamp'])
        stop = float(row[prefix + '/to_timestamp'])
        assert abs((stop - start) * info['fps'] - n) < .01
        video = info['video_path'].format(video_key=VIDEO_KEY,
            chunk_index=row[prefix + '/chunk_index'], file_index=row[prefix + '/file_index'])
        assert (root / video).is_file(), root / video
        episodes.append(dict(episode=eid, task=row['tasks'][0], task_index=tasks.index(row['tasks'][0]),
            frames=n, pairs=max(0, n-HORIZON), global_start=offset, video=video,
            video_start=start, fps=info['fps']))
        offset += n
    assert offset == info['total_frames'] and len(episodes) == info['total_episodes']
    assert len(tasks) == info['total_tasks']
    return episodes


def plan_fingerprint(episodes):
    # Relative paths permit staging the same videos onto node-local storage.
    return hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest()


def label_path(root, episode):
    return Path(root) / f"task_{episode['task_index']:02d}" / f"episode_{episode['episode']:06d}.h5"


def decode_episode_frames(root, episode, first=0, stop=None):
    """Decode exact presentation timestamps, including nonzero shard offsets."""
    stop = episode['frames'] if stop is None else stop
    assert 0 <= first < stop <= episode['frames']
    fps, origin = episode['fps'], episode['video_start']
    expected = first
    with av.open(str(Path(root) / episode['video'])) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        target = origin + first / fps
        if target:
            container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError('Missing video PTS')
            local = int(round((frame.time - origin) * fps))
            if local < first:
                continue
            if local >= stop:
                break
            if local != expected or abs(frame.time - (origin + local/fps)) > .005:
                raise ValueError(f"Episode {episode['episode']} PTS mismatch: {local}, expected {expected}, {frame.time}")
            array = frame.to_ndarray(format='rgb24')
            assert array.shape == (480,640,3), array.shape
            yield local, array
            expected += 1
            if expected == stop:
                return
    if expected != stop:
        raise ValueError(f"Truncated episode {episode['episode']}: {expected} != {stop}")


def decode_frame(root, episode, t):
    return next(decode_episode_frames(root, episode, t, t+1))[1]


def preprocess(array, resolution=416):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio, _balanced_target_shape
    image = _crop_to_supported_aspect_ratio(Image.fromarray(array).convert('RGB'))
    w,h = image.size
    gh,gw = _balanced_target_shape(h/w, resolution, 16)
    return to_tensor(image.resize((gw,gh), Image.Resampling.BICUBIC))


def pair_stream(root, episode, resume=0):
    """Each frame is decoded/preprocessed once, in worker processes."""
    ring = collections.deque(maxlen=HORIZON+1)
    for t,array in decode_episode_frames(root, episode, first=resume):
        ring.append(preprocess(array))
        if len(ring) == HORIZON+1:
            import torch
            yield t-HORIZON, torch.stack([ring[0], ring[-1]])
