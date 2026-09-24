from __future__ import annotations
import json,copy,sys,subprocess,zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from c1.protocol import ROOT,read,validate,jobs,config_for,verify_core,prerequisites,release,source_manifest
from c1.analysis import gm,dataset_gate,aggregate
from c1.numerics import score,compare_arrays,normalized_metric_delta,predict_views,preflight
from c1.runner import spawn,collect
from pi4.io import write
from pi4.learning import model_for,reset_views,general_views
from pi4.data import reset_pairs,general,make
P=read(ROOT/'configs/confirm.json')
def test_frozen_core_exact():assert verify_core()['status']=='PASS'
def test_configuration_frozen():validate(P)
@pytest.mark.parametrize('field,value',[('learning_rate',.004),('width',25),('updates',1601),('difference_weight',.5)])
def test_reject_scientific_changes(field,value):
    p=copy.deepcopy(P);p['fit'][field]=value
    with pytest.raises(ValueError):validate(p)
def test_schedule_complete_and_repeatable():
    j=jobs(P);assert len(j)==45 and len({(x['dataset'],x['method'],x['seed']) for x in j})==45
    assert j==jobs(P)
    assert set(x['dataset'] for x in j)=={'d01','d02','d03'}
def test_data_seed_overlap_rejected():
    p=copy.deepcopy(P);p['datasets'][1]['data_seed']=p['datasets'][0]['data_seed']+1
    with pytest.raises(ValueError):validate(p)
def test_historical_seed_overlap_rejected():
    p=copy.deepcopy(P);p['datasets'][0]['data_seed']=920510
    with pytest.raises(ValueError):validate(p)
def test_old_initialization_rejected():
    p=copy.deepcopy(P);p['initialization_seeds'][0]=727
    with pytest.raises(ValueError):validate(p)
def test_config_propagation():
    c=config_for(P,P['datasets'][1]);assert c['data_seed']==922101 and c['seeds']==[811,821,823] and c['updates']==1600

def synthetic_records(ratio=.5,general_ratio=1.):
    out=[]
    for m in P['methods']:
      for s in P['initialization_seeds']:
        for name in ['ring4','ring8','long8']:
          out.append(dict(method=m,seed=s,case=name,wait=0,kind='general',mse=general_ratio if m=='fixed_behavior' else 1.))
        for name in ['reset_iso','reset_ring4','reset_null']:
          for w in [0,256]:
            out.append(dict(method=m,seed=s,case=name,wait=w,kind='reset',mse=.5,
               difference_error=ratio if m=='fixed_behavior' else 1.,repeat_vs_difference=.5,wrong_ratio=2.))
    return out

def test_gate_positive_and_negative():
    r=dataset_gate(synthetic_records(),P['initialization_seeds'],256,P['decision']);assert r['passed'] and not r['boundary_sensitive']
    assert not dataset_gate(synthetic_records(1.2),P['initialization_seeds'],256,P['decision'])['passed']
    assert not dataset_gate(synthetic_records(.5,1.2),P['initialization_seeds'],256,P['decision'])['passed']
def test_boundary_not_silently_passed():
    r=dataset_gate(synthetic_records(.90001),P['initialization_seeds'],256,P['decision']);assert r['boundary_sensitive']
def test_duplicate_rows_rejected():
    r=synthetic_records();r.append(r[0])
    with pytest.raises(ValueError):dataset_gate(r,P['initialization_seeds'],256,P['decision'])
def test_missing_seed_not_dropped():
    r=[r for r in synthetic_records() if r['seed']!=811]
    with pytest.raises(KeyError):dataset_gate(r,P['initialization_seeds'],256,P['decision'])
@pytest.mark.parametrize('x',[[0,1],[-1,2],[float('nan')],[]])
def test_no_epsilon_masking(x):
    with pytest.raises(ValueError):gm(x)

def populate_results(tmp_path,values,fail_audit=False):
    for d,v in zip(P['datasets'],values):
        path=tmp_path/'datasets'/d['id'];rows=synthetic_records(v)
        write(path/'summary.json',{'records':rows})
        write(path/'audit.json',{'status':'FAIL' if fail_audit else 'PASS','legacy_status':'FAIL','replayed_records':rows})
def test_all_data_required_for_replication(tmp_path):
    populate_results(tmp_path,[.5,.6,.7]);r=aggregate(tmp_path,P)
    assert r['status']=='REPLICATED_WITHIN_SIMULATOR' and r['dataset_pass_count']==3
    # Historical elementwise failure alone is retained, not substituted for the new fidelity gate.
    assert all(d['legacy_elementwise_status']=='FAIL' for d in r['datasets'])
def test_partial_not_called_complete(tmp_path):
    populate_results(tmp_path,[.5,.6,1.1]);assert aggregate(tmp_path,P)['status']=='PARTIAL_REPLICATION'
def test_numerical_failure_inconclusive(tmp_path):
    populate_results(tmp_path,[.5,.6,.7],True);assert aggregate(tmp_path,P)['status']=='INCONCLUSIVE'
def test_missing_data_inconclusive(tmp_path):
    populate_results(tmp_path,[.5,.6]);assert aggregate(tmp_path,P)['status']=='INCONCLUSIVE'
def test_no_evaluation_release_before_all_fits(tmp_path):
    bad=prerequisites(tmp_path,P);assert len(bad)==45
    with pytest.raises(RuntimeError):release(tmp_path,P)
    assert not (tmp_path/'TEST_RELEASE_LOCK.json').exists()
def test_complete_print_does_not_override_timeout(tmp_path):
    r=spawn([sys.executable,'-c','import time;print("COMPLETE",flush=True);time.sleep(2)'],tmp_path/'out.log',.1)
    assert r['status']=='TIMEOUT' and r['timeout_observed']
def test_nonzero_exit_is_error(tmp_path):
    r=spawn([sys.executable,'-c','raise SystemExit(4)'],tmp_path/'out.log',5)
    assert r['status']=='ERROR' and r['exit_code']==4

def test_legacy_element_failure_reported_without_abort():
    p=P['numerics'];a=np.zeros(100);b=a.copy();b[0]=3e-5
    r=compare_arrays(a,b,1.,p);assert r['legacy_bad_elements']==1 and r['pass']
def test_large_array_drift_fails():
    r=compare_arrays(np.zeros(100),np.ones(100)*.01,1.,P['numerics']);assert not r['pass']
def test_nonfinite_rejected():
    r=compare_arrays(np.zeros(2),np.array([1,np.nan]),1.,P['numerics']);assert not r['pass']
def test_independent_scoring_and_null_denominator():
    y=np.zeros((2,2,4,1));p=np.ones_like(y);old={'method':'fixed_behavior','seed':811,'kind':'reset','case':'reset_null','wait':256}
    r=score(old,{'prediction':p,'alternate':p*.5,'wrong':p*2},y,{'reset':2.,'general':1.})
    assert r['mse']==1 and r['nmse']==.5 and r['difference_error'] is None and r['repeat_mse']==.25
    assert normalized_metric_delta(r,r,{'reset':2.,'general':1.})==0
@pytest.mark.parametrize('method',P['methods'])
@pytest.mark.parametrize('kind',['general','reset'])
def test_replay_matches_reference_views(method,kind):
    torch.manual_seed(811);m=model_for(method).eval()
    raw,_=(general if kind=='general' else reset_pairs)(2,1,'isolated',4,4,9281,length=8)
    d={k:torch.from_numpy(v) for k,v in raw.items()}
    with torch.no_grad():
        r=predict_views(m,d,kind,2 if kind=='reset' else 0)
        o=(general_views(m,d)[0] if kind=='general' else reset_views(m,d,2)[0])
    if kind=='general':a=o[:2].numpy();b=o[2:].numpy()
    else:a=o[:,:,0].numpy();b=o[:,:,1].numpy()
    np.testing.assert_allclose(r['prediction'],a,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(r['alternate'],b,rtol=1e-5,atol=1e-5)

def test_collection_excludes_only_regenerable_data(tmp_path):
    run=tmp_path/'run';write(run/'protocol.json',P);write(run/'REPORT_CN.md',{'status':'failed example'})
    (run/'datasets/d01/data').mkdir(parents=True);(run/'datasets/d01/evaluation').mkdir()
    np.savez(run/'datasets/d01/data/train_g2.npz',x=np.zeros(2))
    np.savez(run/'datasets/d01/evaluation/pred.npz',x=np.zeros(2))
    r=collect(run,tmp_path/'return.zip');assert r['mismatches']==[] and r['excluded_data_files']==1
    with zipfile.ZipFile(tmp_path/'return.zip') as z:
        assert 'run/datasets/d01/evaluation/pred.npz' in z.namelist()
        assert 'run/datasets/d01/data/train_g2.npz' not in z.namelist()
    with pytest.raises(FileExistsError):collect(run,tmp_path/'return.zip')

def test_saved_difference_matches_frozen_float32_subtraction():
    # Deliberately use values for which subtract-then-cast differs from cast-then-subtract.
    rng=np.random.default_rng(441);y=rng.normal(size=(4,2,8,1)).astype('float32')
    pred=rng.normal(size=y.shape).astype('float32')
    row={'method':'fixed_behavior','seed':811,'kind':'reset','case':'reset_iso','wait':256}
    r=score(row,{'prediction':pred,'alternate':pred*.7,'wrong':pred*1.2},y,{'reset':1.,'general':1.})
    from pi4.experiment import mse,ratio
    dy=y[:,1]-y[:,0];de=pred[:,1]-pred[:,0]
    energy=float(np.mean(dy.astype('float64')**2))
    assert r['difference_error']==ratio(mse(de,dy),energy)
    assert r['difference_target_energy']==energy

def test_cpu_smoke_does_not_mask_audit_failure(tmp_path):
    p=copy.deepcopy(P);p['stage']='cpu_smoke'
    populate_results(tmp_path,[.5,.6,.7],True)
    assert aggregate(tmp_path,p)['status']=='CPU_ENGINEERING_FAIL'

def test_cpu_smoke_cannot_be_required_to_win_scientific_gate(tmp_path):
    p=copy.deepcopy(P);p['stage']='cpu_smoke'
    populate_results(tmp_path,[1.0001,1.0001,1.0001])
    out=aggregate(tmp_path,p)
    assert out['status']=='CPU_ENGINEERING_ONLY_NOT_SCIENTIFIC'
    assert out['dataset_pass_count']==0
