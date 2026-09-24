from __future__ import annotations
import json,time,sys,random
from pathlib import Path
import numpy as np
import torch
from .io import write,sha,seed_all,load,take
from .learning import METHODS,model_for,objective,general_views,reset_views

@torch.no_grad()
def valid(model,ds,scales):
    model.eval();g=[]
    for d in ds[:2]:
        p,_,_=general_views(model,d);y=torch.cat([d['target']]*2,0)
        g.append(float((p-y).square().mean())/scales['general'])
    d=ds[2];r=[]
    for wait in (0,16):
        p,_,_=reset_views(model,d,wait=wait)
        r.append(float((p-d['target'][:,:,None]).square().mean())/scales['reset'])
    return .5*float(np.mean(g))+.5*float(np.mean(r)),{'general_nmse':float(np.mean(g)),'reset_nmse':float(np.mean(r))}

def fit(config,method,seed,datadir,out,device,started=None):
    start=time.monotonic() if started is None else started
    seed_all(seed);datadir=Path(datadir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (datadir.parent/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('cannot train after evaluation release')
    scales=json.loads((datadir/'SCALES.json').read_text())
    train=[load(datadir/f'train_g{n}.npz',device) for n in (2,4)]+[load(datadir/'train_reset.npz',device)]
    val=[load(datadir/f'val_g{n}.npz',device) for n in (2,4)]+[load(datadir/'val_reset.npz',device)]
    model=model_for(method,config['width'],config['history_width']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    # Same seed gives EXACTLY the same initialization for the two objectives of each family.
    torch.save({'model_config':model.config(),'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},out/'initial.pt')
    rng=np.random.default_rng(seed+177);curve=[];best=float('inf');best_step=None;done=0;step_costs=[]
    def checkpoint(step,parts=None):
        nonlocal best,best_step
        score,vs=valid(model,val,scales)
        if not np.isfinite(score):raise FloatingPointError('nonfinite validation score')
        row={'step':step,'elapsed':time.monotonic()-start,'selection_score':score,'validation':vs,'train':parts}
        if score<best:
            best=score;best_step=step
            torch.save({'config':model.config(),'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'seed':seed,'method':method,'step':step,'validation_score':score},out/'best.pt')
            row['checkpoint_saved_elapsed']=time.monotonic()-start
        curve.append(row);write(out/'curve.json',curve)
    checkpoint(0)
    for i in range(1,config['updates']+1):
        if time.monotonic()-start>=config['process_limit_seconds']-config['cleanup_reserve_seconds']:break
        which=int(rng.integers(0,2)) if i%2 else 2
        d=train[which];size=config['general_batch'] if which<2 else config['pair_batch']
        ix=rng.choice(len(d['target']),size,replace=False);batch=take(d,ix)
        wait=int(rng.choice(config['train_waits'])) if which==2 else 0
        t=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        loss,parts=objective(model,batch,'general' if which<2 else 'reset',METHODS[method][1],scales,config,wait)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        step_costs.append(time.monotonic()-t);done=i
        if i%config['validate_every']==0 or i==config['updates']:
            checkpoint(i,dict(parts,kind='general' if which<2 else 'reset',wait=wait,loss=float(loss.detach()),gradient_norm=float(gn)))
            print(method,seed,i,best,flush=True)
    if curve[-1]['step']!=done:checkpoint(done)
    torch.save({'config':model.config(),'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'step':done},out/'last.pt')
    result={'status':'COMPLETE' if done==config['updates'] else 'BUDGET_STOP','method':method,'family':METHODS[method][0],'seed':seed,'updates':done,
            'best_step':best_step,'validation_score':best,'worker_elapsed':time.monotonic()-start,
            'parameters':sum(p.numel() for p in model.parameters()),'persistent_dimension':model.persistent_dimension,
            'training_loop_seconds':float(sum(step_costs)),'checkpoint_sha256':sha(out/'best.pt'),'initial_sha256':sha(out/'initial.pt')}
    write(out/'result.json',result);return result

def mse(a,b):return float(np.mean((np.asarray(a,dtype='float64')-np.asarray(b,dtype='float64'))**2))
def ratio(a,b):return float(a/b) if np.isfinite(a) and b>1e-15 else None

@torch.no_grad()
def evaluate(config,run,device='cpu'):
    run=Path(run)
    if (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('already released; use audit, not a new selection')
    expected=[f'{m}_s{s}' for m in config['methods'] for s in config['seeds']]
    missing=[]
    for tid in expected:
        p=run/'trials'/tid/'parent.json'
        if not p.exists() or json.loads(p.read_text())['status']!='COMPLETE':missing.append(tid)
    if missing:
        d={'status':'INCONCLUSIVE','reason':'incomplete prerequisite comparisons','missing':missing};write(run/'decision.json',d);return d
    write(run/'TEST_RELEASE_LOCK.json',{'status':'RELEASED_ONCE','scope':'new PI4 simulated held-out inputs; no changes to any PI2/PI3 test lock','unix_time':time.time()})
    scales=json.loads((run/'data/SCALES.json').read_text());rows=[]
    general_cases=['ring4','ring8','long8'];reset_cases=['reset_iso','reset_ring4','reset_null']
    for method in config['methods']:
        for seed in config['seeds']:
            tid=f'{method}_s{seed}';ck=run/'trials'/tid/'best.pt';s=torch.load(ck,map_location='cpu',weights_only=True)
            model=model_for(method,config['width'],config['history_width']).to(device);model.load_state_dict(s['state_dict']);model.eval()
            dest=run/'evaluation'/tid;dest.mkdir(parents=True,exist_ok=True)
            for name in general_cases:
                d=load(run/'data'/f'{name}.npz',device);p,_,_=general_views(model,d);b=len(d['target'])
                p,alt=p[:b].cpu().numpy(),p[b:].cpu().numpy();y=d['target'].cpu().numpy()
                wrong,_,_=general_views(model,d,wrong=True);wrong=wrong[:b].cpu().numpy()
                np.savez_compressed(dest/f'{name}.npz',prediction=p,alternate=alt,wrong=wrong)
                err=mse(p,y);rows.append({'method':method,'seed':seed,'case':name,'kind':'general','wait':0,'mse':err,'nmse':err/scales['general'],
                                        'repeat_mse':mse(p,alt),'wrong_mse':mse(wrong,y),'wrong_ratio':ratio(mse(wrong,y),err)})
            for name in reset_cases:
                d=load(run/'data'/f'{name}.npz',device);y=d['target'].cpu().numpy()
                for wait in config['test_waits']:
                    p,_,_=reset_views(model,d,wait=wait);p=p.cpu().numpy();pa=p[:,:,0];pb=p[:,:,1]
                    wrong,_,_=reset_views(model,d,wait=wait,wrong=True);wrong=wrong.cpu().numpy()[:,:,0]
                    key=f'{name}_w{wait}';np.savez_compressed(dest/f'{key}.npz',prediction=pa,alternate=pb,wrong=wrong)
                    delta=pa[:,1]-pa[:,0];dy=y[:,1]-y[:,0];energy=float(np.mean(dy.astype('float64')**2));err=mse(pa,y)
                    dot=float(np.mean(delta.astype('float64')*dy));dn=float(np.mean(delta.astype('float64')**2))
                    rows.append({'method':method,'seed':seed,'case':name,'kind':'reset','wait':wait,'mse':err,'nmse':err/scales['reset'],
                                 'difference_mse':mse(delta,dy),'difference_target_energy':energy,'difference_error':ratio(mse(delta,dy),energy),
                                 'difference_output_energy':dn,'difference_amplitude_ratio':ratio(dn,energy),
                                 'repeat_mse':mse(pa,pb),'repeat_vs_difference':ratio(mse(pa,pb),energy),
                                 'wrong_mse':mse(wrong,y),'wrong_ratio':ratio(mse(wrong,y),err)})
            print('evaluated',tid,flush=True)
    summary={'protocol':config['protocol'],'status':'EXPLORATORY_COMPLETE','gpu':'EXECUTED' if str(device).startswith('cuda') else 'NOT_RUN',
             'records':rows,'scales':scales,'n_initializations':len(config['seeds']),
             'limits':['one simulator','one data split','new behavioral training distribution','not wall-clock matched','not a new architecture','no hidden theta labels']}
    write(run/'summary.json',summary);return summary
