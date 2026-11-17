"""Download official pretrained weights; never store account tokens in this repo."""
import argparse
import os
from pathlib import Path
import urllib.request
from huggingface_hub import hf_hub_download

parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['baseline','geometry'],default='geometry')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/'checkpoints'
for repo,name in [('black-forest-labs/FLUX.2-klein-base-4B','flux-2-klein-base-4b.safetensors'),('black-forest-labs/FLUX.2-dev','ae.safetensors')]:
    print(hf_hub_download(repo_id=repo,filename=name,local_dir=root/'flux2'/repo.split('/')[-1]))
if args.mode=='geometry':
    target=Path(os.environ.get('DEPTH_PRO_CHECKPOINT',str(root/'depth_pro/depth_pro.pt')))
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.download')
        urllib.request.urlretrieve('https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt',tmp)
        tmp.replace(target)
    print(target)
