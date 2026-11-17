"""Fixed development windows and diffusion noise for short loss-weight comparisons."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from omegaconf import OmegaConf
from .datasets.simple_rgbd import SimpleRGBDDataset

def run_fixed_eval(trainer):
    cfg=trainer.cfg
    model=trainer.accelerator.unwrap_model(trainer.model)
    if not hasattr(trainer,'_fixed_loss_dataset'):
        args=OmegaConf.to_container(cfg.data.train,resolve=True)
        args.pop('_target_');args['tuning_partition']='dev'
        ds=SimpleRGBDDataset(**args)
        selected=[]
        for ei,row in enumerate(ds.episodes):
            base=0 if ei==0 else ds.offsets[ei-1]
            for fraction in (.1,.35,.65,.9):
                selected.append(base+int((len(ds.starts[ei])-1)*fraction))
        trainer._fixed_loss_dataset=ds
        trainer._fixed_loss_indices=selected
    ds=trainer._fixed_loss_dataset
    indices=trainer._fixed_loss_indices
    rank=trainer.accelerator.process_index
    world=trainer.accelerator.num_processes
    sums={};count=0
    model.eval()
    with torch.random.fork_rng(devices=[trainer.accelerator.device]),torch.no_grad():
        for pos in range(rank,len(indices),world):
            torch.manual_seed(2026111600+pos)
            sample=default_collate([ds[indices[pos]]])
            with trainer.accelerator.autocast():
                loss,metrics=model.training_loss(sample)
            assert torch.isfinite(loss),'nonfinite fixed-development loss'
            for name,value in metrics.items():sums[name]=sums.get(name,0.)+float(value)
            count+=1
    keys=sorted(sums)
    values=torch.tensor([count]+[sums[k] for k in keys],device=trainer.accelerator.device,dtype=torch.float64)
    if torch.distributed.is_initialized():torch.distributed.all_reduce(values)
    result={k:float(values[i+1]/values[0]) for i,k in enumerate(keys)}
    result.update(step=trainer.global_step,samples=int(values[0]),
                  evaluation='64 fixed windows from 16 development episodes; diffusion noise fixed per window; development episodes excluded from short-run training; normalization statistics retained from full dataset')
    if trainer.accelerator.is_main_process:
        p=Path(trainer.output_dir)/'fixed_eval.jsonl'
        with p.open('a') as f:f.write(json.dumps(result)+'\n')
        print('[fixed_eval] '+json.dumps(result),flush=True)
    trainer._set_dit_only_train_mode()
    trainer.accelerator.wait_for_everyone()
    return result
