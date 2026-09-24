"""Finite real-path CPU/CUDA comparisons; no BF16/compile timing matrix."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .io import write,load,take,seed_all
from .learning import model_for,objective,METHODS

def relative(a,b):return float(torch.linalg.vector_norm(a-b)/(torch.linalg.vector_norm(a)+1e-8))
def check(config,datadir,device,out):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datadir=Path(datadir);scales=json.loads((datadir/'SCALES.json').read_text());rows=[]
    for method in config['methods']:
        for kind,name in [('general','train_g2'),('reset','train_reset')]:
            seed_all(901);cpu=model_for(method).float();gpu=model_for(method).float().to(device);gpu.load_state_dict(cpu.state_dict())
            d=take(load(datadir/f'{name}.npz'),[0,1]);dg={k:v.to(device) for k,v in d.items()}
            oc=torch.optim.AdamW(cpu.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            og=torch.optim.AdamW(gpu.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            lc,_=objective(cpu,d,kind,METHODS[method][1],scales,config,wait=0)
            lg,_=objective(gpu,dg,kind,METHODS[method][1],scales,config,wait=0)
            lc.backward();lg.backward();cg=torch.cat([p.grad.flatten() for p in cpu.parameters() if p.grad is not None]);gg=torch.cat([p.grad.detach().cpu().flatten() for p in gpu.parameters() if p.grad is not None])
            pc=torch.cat([p.detach().flatten() for p in cpu.parameters()]);pg=torch.cat([p.detach().cpu().flatten() for p in gpu.parameters()])
            oc.step();og.step();uc=torch.cat([p.detach().flatten() for p in cpu.parameters()])-pc;ug=torch.cat([p.detach().cpu().flatten() for p in gpu.parameters()])-pg
            row={'method':method,'kind':kind,'loss_rel':abs(float(lc.detach()-lg.detach().cpu()))/(abs(float(lc.detach()))+1e-8),
                 'gradient_rel':relative(cg,gg),'update_rel':relative(uc,ug)}
            row['pass']=row['loss_rel']<1e-3 and row['gradient_rel']<3e-3 and row['update_rel']<.03
            rows.append(row)
    write(out,{'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','checks':rows,'scope':'finite FP32 training paths on tiny new inputs; not general correctness'})
    if not all(r['pass'] for r in rows):raise RuntimeError('GPU preflight failed; no training launched')
