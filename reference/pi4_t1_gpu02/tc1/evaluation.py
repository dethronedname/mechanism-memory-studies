from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import torch
from pi4.io import write,sha,load
from pi4.learning import general_views,reset_views
from bc1.models import model_for
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
    write(run/'TEST_RELEASE_LOCK.json',{'status':'RELEASED_ONCE','scope':'T1 new held-out data, primary 600-second fit only; earlier experiments untouched','unix_time':time.time()})
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
             'limits':['one simulator','one data split','new behavioral training distribution','matched complete-fit time upper bounds, not exact realized time or equal research cost','not a new architecture','no hidden theta labels']}
    write(run/'summary.json',summary);return summary
