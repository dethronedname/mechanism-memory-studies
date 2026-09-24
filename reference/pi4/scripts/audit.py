#!/usr/bin/env python3
"""Independent replay orchestration. Not an independent implementation of the ODE network."""
from __future__ import annotations
import sys,json,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from pi2.models import Model
from pi4.io import load,write,sha
from pi4.data import make
from pi4.experiment import mse

def jsonread(p):return json.loads(Path(p).read_text())
@torch.no_grad()
def forward(m,h,s,e,k,wait=0):
    z,c=m.initialize(h,s)
    for _ in range(wait):z=m.advance(z,c,torch.zeros(z.shape[:2]),k)
    yy=[]
    for j in range(e.shape[1]):
        z=m.advance(z,c,e[:,j],k);yy.append(m.port(z,c))
    return torch.stack(yy,1).numpy()

def run(runpath,out):
    t=time.monotonic();torch.set_num_threads(1);r=Path(runpath);cfg=jsonread(r/'config.json');summ=jsonread(r/'summary.json')
    maxpred=0.;maxmse=0.;checked=0;selection=[];initial=[];root=Path(__file__).resolve().parents[1]
    source=jsonread(r/'SOURCE_MANIFEST.json');source_bad=[p for p,h in source.items() if sha(root/p)!=h]
    with tempfile.TemporaryDirectory() as td:
        manifest,sc=make(cfg,Path(td)/'data');old=jsonread(r/'data/MANIFEST.json')
        bad=[k for k in old if old[k]['array_hash']!=manifest[k]['array_hash']]
    for method in cfg['methods']:
        for seed in cfg['seeds']:
            tid=f'{method}_s{seed}';d=r/'trials'/tid;curve=jsonread(d/'curve.json');res=jsonread(d/'result.json')
            expected=min(curve,key=lambda x:x['selection_score'])
            assert expected['step']==res['best_step'];assert expected['selection_score']==res['validation_score']
            save=torch.load(d/'best.pt',map_location='cpu',weights_only=True);assert save['validation_score']==expected['selection_score']
            selection.append({'trial':tid,'best_step':expected['step'],'passed':True})
            m=Model(**save['config']);m.load_state_dict(save['state_dict']);m.eval()
            for row in [x for x in summ['records'] if x['method']==method and x['seed']==seed]:
                name=row['case'];wait=row['wait'];data=load(r/'data'/f'{name}.npz')
                filename=name if row['kind']=='general' else f'{name}_w{wait}'
                with np.load(r/'evaluation'/tid/f'{filename}.npz') as z:oldp={k:z[k] for k in z.files}
                if row['kind']=='general':
                    y=data['target'].numpy();a=forward(m,data['history'],data['support_a'],data['external'],data['k'])
                    b=forward(m,data['history'],data['support_b'],data['external'],data['k'])
                    w=forward(m,data['history'],torch.roll(data['support_a'],1,0),data['external'],data['k'])
                else:
                    B,worlds,H,N,C=data['history'].shape
                    def flat(x):return x.reshape(B*worlds,*x.shape[2:])
                    a=forward(m,flat(data['history']),flat(data['support_a']),flat(data['external']),data['k'],wait).reshape(B,worlds,-1,N)
                    b=forward(m,flat(data['history']),flat(data['support_b']),flat(data['external']),data['k'],wait).reshape(B,worlds,-1,N)
                    w=forward(m,flat(data['history']),flat(data['support_a'].flip(1)),flat(data['external']),data['k'],wait).reshape(B,worlds,-1,N)
                    y=data['target'].numpy()
                    dm=mse(oldp['prediction'][:,1]-oldp['prediction'][:,0],y[:,1]-y[:,0]);maxmse=max(maxmse,abs(dm-row['difference_mse']))
                maxmse=max(maxmse,abs(mse(oldp['prediction'],y)-row['mse']),abs(mse(oldp['wrong'],y)-row['wrong_mse']),abs(mse(oldp['prediction'],oldp['alternate'])-row['repeat_mse']))
                for key,pred in [('prediction',a),('alternate',b),('wrong',w)]:
                    err=float(np.max(np.abs(pred-oldp[key])));maxpred=max(maxpred,err)
                    np.testing.assert_allclose(pred,oldp[key],rtol=2e-5,atol=2e-5);checked+=1
            print('audited',tid,flush=True)
    for family in ['fixed','free']:
        for seed in cfg['seeds']:
            a=torch.load(r/'trials'/f'{family}_mse_s{seed}'/'initial.pt',weights_only=True)['state_dict']
            b=torch.load(r/'trials'/f'{family}_behavior_s{seed}'/'initial.pt',weights_only=True)['state_dict']
            err=max(float((a[k]-b[k]).abs().max()) for k in a)
            initial.append({'family':family,'seed':seed,'max_parameter_difference':err});assert err==0
    result={'status':'PASS' if not source_bad and not bad and maxmse<1e-12 else 'FAIL','source_files':len(source),'source_mismatches':source_bad,
            'regenerated_data_cases':len(old),'data_mismatches':bad,'checked_prediction_arrays':checked,
            'max_cpu_forward_abs_difference':maxpred,'max_saved_metric_recompute_difference':maxmse,
            'selection_checks':selection,'matched_initialization_checks':initial,'seconds':time.monotonic()-t,
            'scope':'independent scoring/replay orchestration, same underlying model code; no training and no GPU'}
    write(out,result);return result
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
