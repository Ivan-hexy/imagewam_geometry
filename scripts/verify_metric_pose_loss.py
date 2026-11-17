import json, math
from pathlib import Path
import torch
from torch import nn
from imagewam.models.backbones.pose_objective import rotation_angle, rotation_residual, translation_residual
from imagewam.models.backbones.global_depth import GlobalDepthRegisters, rotation_matrix
torch.set_num_threads(2)
cfg={'pose_loss_type':'metric_normalized','pose_rotation_scale_rad':.02,'pose_translation_scale_m':.1,'lambda_pose':.1,'lambda_xyz':.5}
tests=[]
for angle in [0,1e-6,.01,.3,math.pi-1e-5,math.pi]:
    a=torch.tensor(angle,dtype=torch.float32,requires_grad=True)
    z=a*0;o=z+1;c=a.cos();s=a.sin()
    R=torch.stack([c,-s,z,s,c,z,z,z,o]).reshape(3,3)
    val=rotation_angle(R,torch.eye(3));val.backward()
    assert abs(val.item()-angle)<1e-6 and torch.isfinite(a.grad)
    tests.append({'angle':angle,'loss':float(val),'gradient':float(a.grad)})
x=torch.tensor([[.1,-.2,.3]],requires_grad=True);val=translation_residual(x,torch.zeros_like(x),cfg).sum();val.backward()
assert torch.allclose(x.grad,x.detach()/x.detach().norm(dim=-1,keepdim=True)/.1)
zero=torch.zeros(1,3,requires_grad=True);translation_residual(zero,zero.detach(),cfg).sum().backward();assert zero.grad.eq(0).all()
geo=GlobalDepthRegisters.__new__(GlobalDepthRegisters);nn.Module.__init__(geo)
geo.cfg=cfg;geo.predict_translation=True
geo.rotation_head=nn.Sequential(nn.Linear(9,6,bias=False));geo.translation_head=nn.Sequential(nn.Linear(9,3,bias=False))
with torch.no_grad():
    geo.rotation_head[0].weight.zero_();geo.rotation_head[0].weight[:,:6].copy_(torch.eye(6))
    geo.translation_head[0].weight.zero_();geo.translation_head[0].weight[:,6:].copy_(torch.eye(3))
geo.decode=lambda snapshots,K:torch.ones(2,2,2)
pose=torch.tensor([1.,0,0,0,1,0,.1,.02,-.03]).repeat(2,3,1).requires_grad_()
mask=torch.tensor([[True,False,False],[True,True,False]])
R=torch.eye(3).repeat(2,3,1,1);R[~mask]=float('nan')
t=torch.zeros(2,3,3);t[~mask]=float('nan')
sample={'geometry_K':torch.eye(3).repeat(2,1,1),'geometry_depth':torch.ones(2,2,2),'geometry_valid':torch.ones(2,2,2,dtype=torch.bool),'geometry_rotation':R,'geometry_rotation_valid':mask,'geometry_translation':t,'geometry_translation_valid':mask}
total,metrics=geo.losses([],pose,sample);total.backward()
assert torch.isfinite(total) and torch.isfinite(pose.grad).all() and pose.grad[~mask].eq(0).all()
assert abs(total.item()-.1*.5*math.sqrt(.1**2+.02**2+.03**2)/.1)<1e-6
assert torch.equal(rotation_residual(R.nan_to_num(),R.nan_to_num()+.01,{}),(R.nan_to_num()-(R.nan_to_num()+.01)).square().sum((-1,-2))/8)
legacy_t=translation_residual(x.detach(),torch.zeros_like(x),{})
assert torch.equal(legacy_t,torch.nn.functional.smooth_l1_loss(x.detach(),torch.zeros_like(x),reduction='none',beta=.05).mean(-1))
for bad in [0.,-1.,float('nan')]:
    try:translation_residual(x,x,dict(cfg,pose_translation_scale_m=bad))
    except ValueError:pass
    else:raise AssertionError('invalid scale accepted')
report={'passed':True,'rotation_extremes':tests,'translation_analytical_gradient':True,'invalid_nan_labels_masked':True,'invalid_gradients_zero':True,'shared_weight_formula_checked':True}
Path('loss_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
