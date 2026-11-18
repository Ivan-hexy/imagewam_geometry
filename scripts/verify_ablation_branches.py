"""CPU checks of structural ablations, masks, gradients and 25-layer KV reuse.

Uses a narrow transformer and real Depth Pro decoder weights, not a full-size
GPU training/convergence test. No source datasets are changed.
"""
import argparse,importlib.util,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(os.environ.get('FLUX2_SRC',ROOT/'third_party/flux2'))/'src'))
from imagewam.models.backbones.global_depth import GlobalDepthRegisters
from imagewam.models.backbones.aligned_geometry_forward import aligned_forward
from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert
from imagewam.models.backbones.action_dit_flux2 import ActionDiTFlux2
from imagewam.models.backbones.mot import MoT
from imagewam.models.backbones.imagewam_global_depth import ImageWAMGlobalDepth
from flux2.model import Flux2,Klein4BParams

parser=argparse.ArgumentParser();parser.add_argument('--legacy-global-depth');args=parser.parse_args()
torch.set_num_threads(2)
cfg=dict(grid=[11,20],resolution=[32,64],horizon=32,lambda_xyz=.5,lambda_rotation=5.,predict_translation=False,
    depth_pro_source=os.environ.get('DEPTH_PRO_SRC',str(ROOT/'third_party/ml-depth-pro')),
    depth_pro_checkpoint=os.environ.get('DEPTH_PRO_CHECKPOINT',str(ROOT/'checkpoints/depth_pro/depth_pro.pt')))
torch.manual_seed(7);both=GlobalDepthRegisters(cfg,128,64)
pose=GlobalDepthRegisters(dict(cfg,depth_enabled=False,depth_pro_checkpoint='deliberately_missing'),128,64)
depth=GlobalDepthRegisters(dict(cfg,pose_enabled=False),128,64)
assert pose.global_tokens.shape[1]==0 and not hasattr(pose,'decoder') and not hasattr(pose,'head')
assert depth.pose_tokens.shape[1]==0 and not hasattr(depth,'rotation_head')
assert both.pose_tokens.shape[1]==32 and both.global_tokens.shape[1]==220
assert not any('translation_head' in n for m in [both,pose,depth] for n,_ in m.named_parameters())
z=torch.randn(1,32,128,requires_grad=True)
snap=[torch.randn(1,220,64,requires_grad=True) for _ in range(5)]
rv=torch.zeros(1,32,dtype=torch.bool);rv[:,:5]=True
rot=torch.eye(3).repeat(1,32,1,1);rot[:,5:]=float('nan')
sample={'geometry_K':torch.tensor([[[50.,0,32],[0,50,16],[0,0,1.]]]),'geometry_depth':torch.ones(1,32,64),
    'geometry_valid':torch.ones(1,32,64,dtype=torch.bool),'geometry_rotation':rot,'geometry_rotation_valid':rv}
legacy_report=None
if args.legacy_global_depth:
    spec=importlib.util.spec_from_file_location('imagewam.models.backbones._legacy_global_depth',args.legacy_global_depth)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    old=legacy.GlobalDepthRegisters(cfg,128,64);old.load_state_dict(both.state_dict(),strict=True)
    old_loss,_=old.losses(snap,z,sample);new_loss,_=both.losses(snap,z,sample)
    old_g=torch.autograd.grad(old_loss,[z,*snap],retain_graph=True)
    new_g=torch.autograd.grad(new_loss,[z,*snap],retain_graph=True)
    assert torch.equal(old_loss,new_loss)
    assert all(torch.equal(a,b) for a,b in zip(old_g,new_g))
    legacy_report={'identical_loss':True,'identical_input_gradients':True}
    del old,old_loss,new_loss,old_g,new_g
for module,inputs in [(both,sample),(pose,{k:v for k,v in sample.items() if 'rotation' in k}),(depth,{k:v for k,v in sample.items() if 'rotation' not in k})]:
    module.zero_grad(set_to_none=True);z.grad=None
    loss,metrics=module.losses(snap if module.depth_enabled else [],z if module.pose_enabled else z[:,:0],inputs)
    assert torch.isfinite(loss);loss.backward()
    if module.pose_enabled:
        assert z.grad is not None and z.grad[:,5:].eq(0).all()
        assert metrics['rotation_valid_steps']==5
    if not module.pose_enabled:assert metrics['loss_relative_rotation']==0
    if not module.depth_enabled:assert metrics['loss_global_xyz']==0
del both,pose,depth,snap,z

torch.manual_seed(7)
v=Flux2VideoExpert(Flux2(Klein4BParams(hidden_size=128,num_heads=1,context_in_dim=64,mlp_ratio=2.)),'test')
a=ActionDiTFlux2(action_dim=36,hidden_dim=64,num_heads=1,attn_head_dim=128,mlp_ratio=2.)
m=ImageWAMGlobalDepth(v,a,MoT({'video':v,'action':a},mot_checkpoint_mixed_attn=True),vae=torch.nn.Identity(),text_dim=64,device='cpu',stack='flux2')
ref=torch.randn(1,8,128);future=torch.randn_like(ref);txt=torch.randn(1,3,64);act=torch.randn(1,32,36)
ids=v.build_img_ids(1,2,4,time_value=10.,device='cpu',dtype=torch.float32);targetids=ids.clone();targetids[:,:,0]=0
mask=torch.tensor([[1,1,0]],dtype=torch.bool)
inputs=dict(ref_image_latents=ref,ref_img_ids=ids,text_hidden_states=txt,text_attention_mask=mask)
def pre(f=future,x=act):
    return v.pre_dit(f,torch.tensor([.7]),txt,mask,ref,targetids,ids),m._make_clean_pre(inputs),a.pre_dit(x,torch.tensor([.4]))
results={}
for name,np_,ng in [('baseline',0,0),('only_pose',32,0),('only_depth',0,8),('both',32,8)]:
    geo=torch.nn.Module();geo.pose_tokens=torch.nn.Parameter(torch.randn(1,np_,128)*.02);geo.global_tokens=torch.nn.Parameter(torch.randn(1,ng,64)*.02)
    geo.grid=(2,4);geo.snapshot_layers=(4,9,14,19,24) if ng else ();m.geometry_supervision=geo
    with torch.no_grad():
        r=aligned_forward(m,*pre(),return_cache=True)
        changed=aligned_forward(m,*pre(future+20,act-20))
        assert torch.allclose(r['action'],aligned_forward(m,*pre(future+20,act))['action'],atol=3e-5)
        assert torch.allclose(r['pose'],changed['pose'],atol=3e-5)
        assert all(torch.allclose(x,y,atol=3e-5) for x,y in zip(r['snapshots'],changed['snapshots']))
        assert len(r['cache'])==25
        ap=pre()[2];hidden=ap['tokens'];pe=v.transformer.pe_embedder(ap['ids'])
        cache_mask=torch.cat((r['cache_valid'],torch.ones(1,32,dtype=torch.bool)),1)[:,None,:].expand(-1,32,-1)
        for i,(block,cache) in enumerate(zip(a.blocks,r['cache'])):
            st=block.prepare_qkv(hidden,pe,ap['t_mod']['double_img' if i<5 else 'single'])
            mixed=m.mot._mixed_attention(st['q'],torch.cat((cache['k'],st['k']),1),torch.cat((cache['v'],st['v']),1),cache_mask)
            hidden=block.apply_post(mixed,st)
        assert torch.allclose(hidden,r['action'],atol=3e-5,rtol=3e-5)
        if name=='baseline':
            vp,cp,ap=pre();amask=m._build_mot_attention_mask_flux2(1,3,8,8,32,torch.device('cpu'),mask)
            native=m.mot(embeds_all={'video':vp['tokens'],'action':ap['tokens']},attention_mask=amask,freqs_all={'video':vp['freqs']},context_all={'video':None,'action':{'ids':ap['ids']}},t_mod_all={'video':vp['t_mod'],'action':ap['t_mod']})
            assert torch.allclose(native['action'],r['action'],atol=3e-5,rtol=3e-5)
            assert torch.allclose(native['video']['img'],r['video']['img'],atol=3e-5,rtol=3e-5)
    m.zero_grad(set_to_none=True);r=aligned_forward(m,*pre());loss=r['action'].square().mean()
    if np_:loss=loss+r['pose'].square().mean()
    if ng:loss=loss+sum(s.square().mean() for s in r['snapshots'])
    loss.backward()
    if np_:assert geo.pose_tokens.grad is not None and geo.pose_tokens.grad.norm()>0
    if ng:assert geo.global_tokens.grad is not None and geo.global_tokens.grad.norm()>0
    results[name]={'finite_backward':True,'all_25_layer_cache_parity':True,'no_future_leak':True}
print(json.dumps({'passed':True,'actual_constructor_and_depth_decoder_checked':True,'masked_rotation_gradients_zero':True,'both_legacy_parity':legacy_report,'narrow_transformer':results,'note':'CPU FP32 narrow 25-layer test; no full-size GPU convergence claim'},indent=2))
