"""Compare fused AdamW against scalar AdamW on identical fixed gradients."""
import json
from pathlib import Path
import torch
torch.set_num_threads(2)
device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
a=torch.nn.Parameter(torch.randn(4096,device=device))
b=torch.nn.Parameter(a.detach().clone())
oa=torch.optim.AdamW([a],lr=2.5e-5,weight_decay=.01,betas=(.9,.95),foreach=False)
ob=torch.optim.AdamW([b],lr=2.5e-5,weight_decay=.01,betas=(.9,.95),foreach=False,fused=True)
for i in range(20):
    grad=torch.randn_like(a)
    a.grad=grad;b.grad=grad.clone()
    oa.step();ob.step();oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
err=float((a-b).abs().max())
assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
for key in ['exp_avg','exp_avg_sq']:
    assert torch.allclose(oa.state[a][key],ob.state[b][key],atol=1e-6,rtol=1e-6)
result={'passed':True,'device':device,'updates':20,'max_parameter_difference':err,'torch':torch.__version__}
Path('fused_optimizer_smoke_'+device+'.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
