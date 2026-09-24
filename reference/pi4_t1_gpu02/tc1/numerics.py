"""Predeclared numerical checks. Legacy elementwise verdict is always retained.
No cutoff changes at runtime; do not silently drop the first failing prediction.
"""
from __future__ import annotations
import copy, time
from pathlib import Path
import numpy as np
import torch
from pi4.io import load,take,write,seed_all,sha
from pi4.learning import objective
from bc1.models import model_for,METHODS
from .protocol import read,source_manifest

def cast(d,device,dtype):
    return {k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in d.items()}
def relative(a,b):
    a=a.double();b=b.double()
    return float(torch.linalg.vector_norm(a-b)/(torch.linalg.vector_norm(a)+1e-8))
def flat_params(m):return torch.cat([p.detach().cpu().flatten() for p in m.parameters()])
def training_probe(base,data,kind,behavior,scales,c,device,dtype,wait):
    m=copy.deepcopy(base).to(device=device,dtype=dtype).train();d=cast(data,device,dtype)
    opt=torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    before=flat_params(m);loss,_=objective(m,d,kind,behavior,scales,c,wait)
    loss.backward()
    grad=torch.cat([p.grad.detach().cpu().flatten() for p in m.parameters() if p.grad is not None])
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    moments={}
    for key in ('exp_avg','exp_avg_sq'):
        moments[key]=torch.cat([opt.state[p][key].detach().cpu().flatten() for p in m.parameters() if key in opt.state[p]])
    return {'loss':loss.detach().cpu().reshape(1),'gradient':grad,'update':flat_params(m)-before,**moments}

def preflight(c,datadir,device,out,allow_cpu_test=False):
    if not allow_cpu_test and (device!='cuda:0' or not torch.cuda.is_available()):
        raise RuntimeError('CUDA required; no fallback')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scales=read(Path(datadir)/'SCALES.json');rows=[]
    for method in c['methods']:
        for kind,name,wait in [('general','train_g2',0),('reset','train_reset',0),('reset','train_reset',16)]:
            seed_all(911);base=model_for(method,c['width'],c['history_width']).float()
            d=take(load(Path(datadir)/(name+'.npz')),[0,1])
            a=training_probe(base,d,kind,METHODS[method][1],scales,c,'cpu',torch.float32,wait)
            b=training_probe(base,d,kind,METHODS[method][1],scales,c,device,torch.float32,wait)
            r=training_probe(base,d,kind,METHODS[method][1],scales,c,'cpu',torch.float64,wait)
            pair={label:{k:relative(ref[k],test[k]) for k in ref}
                  for label,ref,test in [('gpu32_vs_cpu32',a,b),('gpu32_vs_cpu64',r,b)]}
            limits={'loss':1e-3,'gradient':3e-3,'update':.03,'exp_avg':3e-3,'exp_avg_sq':.006}
            ok=all(np.isfinite(v) and v<=limits[k] for sub in pair.values() for k,v in sub.items())
            rows.append({'method':method,'kind':kind,'wait':wait,'relative_errors':pair,'pass':ok})
    result={'status':'PASS' if all(x['pass'] for x in rows) else 'FAIL','checks':rows,
        'actual_device':device,'scope':'Finite FP32 training-path triplets with clipped AdamW; CPU64 reference is not a GPU performance result.'}
    write(out,result)
    if result['status']!='PASS':raise RuntimeError('Numerical preflight failed; do not start fitting')
    return result

@torch.no_grad()
def forward(m,h,s,e,k,wait=0):
    """Direct unbatched-view replay; independent orchestration, same model formulas."""
    z,c=m.initialize(h,s)
    for _ in range(wait):z=m.advance(z,c,z.new_zeros(z.shape[:2]),k)
    ys=[]
    for j in range(e.shape[1]):
        z=m.advance(z,c,e[:,j],k);ys.append(m.port(z,c))
    return torch.stack(ys,1).detach().cpu().numpy()

@torch.no_grad()
def predict_views(m,d,kind,wait=0):
    if kind=='general':
        h,e,k=d['history'],d['external'],d['k'];a,b=d['support_a'],d['support_b']
        return {'prediction':forward(m,h,a,e,k),'alternate':forward(m,h,b,e,k),
                'wrong':forward(m,h,torch.roll(a,1,0),e,k)}
    batch,worlds,_,nodes,_=d['history'].shape
    def flat(x):return x.reshape(batch*worlds,*x.shape[2:])
    h,e,k=flat(d['history']),flat(d['external']),d['k']
    ans={}
    for key,support in [('prediction',d['support_a']),('alternate',d['support_b']),('wrong',d['support_a'].flip(1))]:
        ans[key]=forward(m,h,flat(support),e,k,wait).reshape(batch,worlds,-1,nodes)
    return ans

def square(x):return float(np.mean(np.asarray(x,dtype=np.float64)**2))
def divide(a,b):return float(a/b) if np.isfinite(a) and b>1e-15 else None

def score(row,pred,y,scales):
    """Score without calling pi4.evaluate: raw metrics regenerated in float64."""
    raw_p=np.asarray(pred['prediction']);raw_y=np.asarray(y)
    p,b,w=[np.asarray(pred[k],dtype=np.float64) for k in ('prediction','alternate','wrong')]
    y=np.asarray(y,dtype=np.float64);err=square(p-y);wrong=square(w-y);rep=square(p-b)
    out={k:row[k] for k in ('method','seed','case','kind','wait')}
    out.update(mse=err,nmse=err/scales['general' if row['kind']=='general' else 'reset'],
               repeat_mse=rep,wrong_mse=wrong,wrong_ratio=divide(wrong,err))
    if row['kind']=='reset':
        # Preserve frozen PI4 subtraction order: subtract in stored array dtype,
        # then aggregate in float64. Changing this changes the historical metric.
        delta=(raw_p[:,1]-raw_p[:,0]).astype(np.float64)
        dy=(raw_y[:,1]-raw_y[:,0]).astype(np.float64)
        energy=square(dy);de=square(delta-dy)
        out.update(difference_mse=de,difference_target_energy=energy,difference_error=divide(de,energy),
          difference_output_energy=square(delta),difference_amplitude_ratio=divide(square(delta),energy),
          repeat_vs_difference=divide(rep,energy))
    return out

def compare_arrays(old,new,scale,policy):
    a=np.asarray(old,dtype=np.float64);b=np.asarray(new,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'pass':False,'error':'shape mismatch or nonfinite','legacy_bad_elements':int(a.size)}
    err=np.abs(a-b);nr=float(np.sqrt(np.mean(err**2)/max(scale,1e-6)));nm=float(err.max()/np.sqrt(max(scale,1e-6)))
    bad=int(np.count_nonzero(err>policy['legacy_atol']+policy['legacy_rtol']*np.abs(a)))
    return {'pass':nr<=policy['normalized_rms_limit'] and nm<=policy['normalized_max_limit'],
        'legacy_bad_elements':bad,'elements':a.size,'max_abs_difference':float(err.max()),
        'train_scale_normalized_rms':nr,'train_scale_normalized_max':nm}

def normalized_metric_delta(old,new,scales):
    base=scales['general' if old['kind']=='general' else 'reset']
    ds=[abs(old[k]-new[k])/base for k in ('mse','repeat_mse','wrong_mse')]
    if old['kind']=='reset':
        ds.extend(abs(old[k]-new[k])/base for k in ('difference_mse','difference_output_energy'))
        if old['difference_error'] is not None and new['difference_error'] is not None:
            ds.append(abs(old['difference_error']-new['difference_error']))
    return max(ds)

@torch.no_grad()
def audit_dataset(path:Path,policy:dict,source_root:Path,config:dict):
    start=time.monotonic();torch.set_num_threads(1)
    summ=read(path/'summary.json');scales=read(path/'data/SCALES.json');errors=[]
    from .protocol import check_trial
    protocol=read(path.parents[1]/'protocol.json')
    original=read(path.parent.parent/'SOURCE_MANIFEST.json')
    current=source_manifest();changed=[k for k,v in original.items() if current.get(k)!=v]
    if changed:errors.append({'reason':'source changed','files':changed})
    # Independently regenerate data with the unchanged simulator and compare content hashes.
    from tempfile import TemporaryDirectory
    from pi4.data import make
    with TemporaryDirectory() as tmp:
        mf,ss=make(config,Path(tmp)/'data')
    old=read(path/'data/MANIFEST.json');data_bad=[k for k in old if mf[k]['array_hash']!=old[k]['array_hash']]
    if data_bad or ss!=scales:errors.append({'reason':'data/scales mismatch','cases':data_bad})
    records=[];arrays=[];probes=[];selections=[];inits=[];worst_metric=0.;worst_saved=0.
    for method in config['methods']:
      for seed in config['seeds']:
        tid=f'{method}_s{seed}';td=path/'trials'/tid
        budget_error=check_trial(td,protocol)
        if budget_error:errors.append({'trial':tid,'reason':'Time/selection audit: '+budget_error})
        try:
            ck=torch.load(td/'best.pt',map_location='cpu',weights_only=True)
            curve=read(td/'curve.json');result=read(td/'result.json')
            from .budget import choose_checkpoint
            best=choose_checkpoint(curve,config['time_policy']['primary_seconds'])
            sel_ok=(best['step']==result['best_step']==ck['step'] and best['selection_score']==result['validation_score']==ck['validation_score'] and sha(td/'best.pt')==result['checkpoint_sha256'])
            m=model_for(method,config['width'],config['history_width']);m.load_state_dict(ck['state_dict']);m.eval()
            from bc1.experiment import valid
            vs,_=valid(m,[load(path/'data'/f'{n}.npz') for n in ('val_g2','val_g4','val_reset')],scales)
            vd=abs(vs-result['validation_score']);sel_ok=sel_ok and vd<=policy['validation_score_limit']
            selections.append({'trial':tid,'pass':sel_ok,'best_step':best['step'],'validation_delta':vd})
            if not sel_ok:errors.append({'trial':tid,'reason':'selection/validation/hash mismatch'})
            rows=[r for r in summ['records'] if r['method']==method and r['seed']==seed]
            if len(rows)!=3+3*len(config['test_waits']):raise ValueError('missing/duplicate scenario rows')
            for row in rows:
                name=row['case'];wait=row['wait'];kind=row['kind'];d=load(path/'data'/f'{name}.npz')
                key=name if kind=='general' else f'{name}_w{wait}'
                with np.load(path/'evaluation'/tid/(key+'.npz'),allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
                replay=predict_views(m,d,kind,wait)
                srow=score(row,saved,d['target'].numpy(),scales);rrow=score(row,replay,d['target'].numpy(),scales);records.append(rrow)
                # Recompute all saved scalar scores, including ratio diagnostics, without prediction rerun.
                sd=0.
                for k,v in srow.items():
                    if k in ('method','seed','case','kind','wait'):continue
                    ov=row.get(k)
                    if v is None or ov is None:
                        if v!=ov:sd=float('inf')
                    else:sd=max(sd,abs(v-ov)/(1+abs(ov)))
                if not np.isfinite(sd) or sd>policy['saved_metric_recompute_limit']:
                    errors.append({'trial':tid,'case':key,'reason':'saved metrics mismatch'})
                else:worst_saved=max(worst_saved,sd)
                md=normalized_metric_delta(srow,rrow,scales);worst_metric=max(worst_metric,md)
                if md>policy['metric_normalized_abs_limit']:errors.append({'trial':tid,'case':key,'reason':'prediction metrics diverged','delta':md})
                scale=scales['general' if kind=='general' else 'reset']
                for k in ('prediction','alternate','wrong'):
                    comp=compare_arrays(saved[k],replay[k],scale,policy);arrays.append({'trial':tid,'case':key,'array':k,**comp})
                    if not comp['pass']:errors.append({'trial':tid,'case':key,'array':k,'reason':'normalized prediction drift'})
                if name=='reset_iso' and wait==max(config['test_waits']):
                    # Fixed finite FP64 spot check for EVERY checkpoint, not just a winning model.
                    ix=list(range(min(policy['fp64_probe_rows'],len(d['target']))));dd=cast(take(d,ix),'cpu',torch.float64)
                    m64=copy.deepcopy(m).double();fp64=predict_views(m64,dd,kind,wait)['prediction']
                    # Compare to matching rows of the original full-batch GPU result.
                    pc=compare_arrays(saved['prediction'][ix],fp64,scale,policy)
                    qrow=score(row,{k:saved[k][ix] for k in saved},d['target'].numpy()[ix],scales)
                    pro={'trial':tid,'case':key,'rows':len(ix),**pc}
                    probes.append(pro)
                    if not pc['pass']:errors.append({'trial':tid,'reason':'FP64 spot-check failed'})
        except Exception as exc:
            errors.append({'trial':tid,'reason':repr(exc)})
        # Continue to all models; never make one failing element erase other evidence.
        write(path/'AUDIT_PROGRESS.json',{'trials_seen':len(selections),'errors':errors,'arrays_seen':len(arrays)})
    for family in ():  # no MSE/behavior factorial in this B1 study
      for seed in config['seeds']:
        try:
            a=torch.load(path/'trials'/f'{family}_mse_s{seed}'/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
            b=torch.load(path/'trials'/f'{family}_behavior_s{seed}'/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
            ok=set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
            inits.append({'family':family,'seed':seed,'pass':ok})
            if not ok:errors.append({'reason':'paired initializations differ','family':family,'seed':seed})
        except Exception as exc:errors.append({'reason':'initialization check error '+repr(exc)})
    bad_elements=sum(r['legacy_bad_elements'] for r in arrays)
    out={'status':'PASS' if not errors else 'FAIL','legacy_status':'FAIL' if bad_elements else 'PASS',
       'legacy_bad_elements':bad_elements,'policy':policy,'errors':errors,'arrays':arrays,'fp64_spot_checks':probes,
       'selections':selections,'paired_initializations':inits,'replayed_records':records,
       'max_saved_metric_scaled_difference':worst_saved,'max_normalized_metric_delta':worst_metric,
       'elapsed_seconds':time.monotonic()-start,
       'scope':'Complete CPU replay, independent metric scoring, same underlying model implementation; fixed small CPU64 probes. Not an independent network implementation.'}
    write(path/'audit.json',out);return out
