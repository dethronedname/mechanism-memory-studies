from __future__ import annotations
import copy,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from bc1.models import model_for,METHODS,NEW_METHODS
from bc1.protocol import read,ROOT,validate,development_jobs,confirmation_jobs,verify_core,select_development,prerequisites
from pi4.data import general,reset_pairs,make
from pi4.io import write,load,sha
from pi4.learning import general_views,reset_views,objective

torch.set_num_threads(1)
CFG=read(ROOT/'configs/challenge.json')['fit']

def tensor(d):return {k:torch.from_numpy(v) for k,v in d.items()}

@pytest.mark.parametrize('m',list(METHODS))
@pytest.mark.parametrize('kind',['general','reset'])
def test_all_real_objective_paths(m,kind):
    torch.manual_seed(133)
    model=model_for(m).double()
    fun=general if kind=='general' else reset_pairs
    d,_=fun(2,2,'chain',4,4,142,length=8)
    d={k:v.double() if v.is_floating_point() else v for k,v in tensor(d).items()}
    loss,_=objective(model,d,kind,True,dict(general=1,reset=1,difference=.2),CFG,wait=2 if kind=='reset' else 0)
    loss.backward();assert torch.isfinite(loss)
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert model.context_head[0].weight.grad.abs().sum()>0
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-6)
    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())

@pytest.mark.parametrize('m',NEW_METHODS)
def test_context_persists_and_affects_every_step(m):
    torch.manual_seed(22);model=model_for(m).double()
    z=torch.randn(2,3,model.d,dtype=torch.double);c=torch.randn(2,3,model.cdim,dtype=torch.double,requires_grad=True)
    keep=c.detach().clone();k=torch.tensor([[0.,.1,0.],[-.1,0.,.2],[0.,-.2,0.]],dtype=torch.double)
    for _ in range(6):z=model.advance(z,c,z.new_zeros(2,3),k)
    g=torch.autograd.grad(model.port(z,c).square().sum(),c)[0]
    assert g.abs().sum()>1e-9;torch.testing.assert_close(c.detach(),keep,rtol=0,atol=0)

@pytest.mark.parametrize('m',NEW_METHODS)
def test_support_order_and_node_permutation(m):
    torch.manual_seed(36);model=model_for(m).double()
    d,_=general(2,4,'ring',4,4,245,length=8);d={k:v.double() if v.is_floating_point() else v for k,v in tensor(d).items()}
    with torch.no_grad():p,_,_=general_views(model,d)
    q=dict(d);q['support_a']=d['support_a'].flip(2);q['support_b']=d['support_b'].flip(2)
    with torch.no_grad():pp,_,_=general_views(model,q)
    torch.testing.assert_close(p,pp,rtol=1e-11,atol=1e-11)
    perm=torch.tensor([2,0,3,1]);q=dict(d)
    for key in ('support_a','support_b'):q[key]=d[key][:,perm]
    for key in ('history','external'):q[key]=d[key][:,:,perm]
    q['k']=d['k'][perm][:,perm]
    with torch.no_grad():qq,_,_=general_views(model,q)
    torch.testing.assert_close(qq,p[:,:,perm],rtol=1e-10,atol=1e-10)

@pytest.mark.parametrize('m',NEW_METHODS)
def test_no_target_access_or_teacher_forcing(m):
    torch.manual_seed(54);model=model_for(m)
    d,_=reset_pairs(2,1,'isolated',5,4,721,length=8);d=tensor(d)
    with torch.no_grad():a,_,_=reset_views(model,d,wait=3)
    d['target'].fill_(1e6);d['y0'].fill_(-1e6)
    with torch.no_grad():b,_,_=reset_views(model,d,wait=3)
    torch.testing.assert_close(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('m',list(METHODS))
def test_save_load(m,tmp_path):
    torch.manual_seed(99);a=model_for(m)
    d,_=general(2,2,'chain',4,4,510,length=8);d=tensor(d)
    with torch.no_grad():p,_,_=general_views(a,d)
    torch.save({'state_dict':a.state_dict(),'method':m,'config':a.config()},tmp_path/'x.pt')
    ck=torch.load(tmp_path/'x.pt',weights_only=True);b=model_for(ck['method']);b.load_state_dict(ck['state_dict'])
    with torch.no_grad():q,_,_=general_views(b,d)
    torch.testing.assert_close(p,q,rtol=0,atol=0)

@pytest.mark.parametrize('m',NEW_METHODS)
def test_double_input_gradcheck(m):
    torch.manual_seed(84);model=model_for(m).double()
    z=torch.randn(1,2,model.d,dtype=torch.double,requires_grad=True)*.1
    c=torch.randn(1,2,model.cdim,dtype=torch.double,requires_grad=True)*.1
    u=torch.randn(1,2,dtype=torch.double,requires_grad=True)*.1
    k=torch.tensor([[0.,.1],[-.1,0.]],dtype=torch.double)
    assert torch.autograd.gradcheck(lambda a,b,e:model.advance(a,b,e,k),(z,c,u),fast_mode=True,atol=1e-5,rtol=1e-3)

@pytest.mark.parametrize('m,expected',[('fixed_behavior',4),('free_behavior',4),('gru4_behavior',4),('gru32_behavior',32),('graph32_behavior',32),('node32_behavior',32)])
def test_persistent_memory_definition(m,expected):
    x=model_for(m);assert x.persistent_dimension==expected

@pytest.mark.parametrize('m',['fixed_behavior','free_behavior'])
def test_candidate_and_old_control_identical(m):
    from pi4.learning import model_for as old
    torch.manual_seed(177);a=old(m);torch.manual_seed(177);b=model_for(m)
    assert a.config()==b.config()
    for k in a.state_dict():torch.testing.assert_close(a.state_dict()[k],b.state_dict()[k],atol=0,rtol=0)
    assert verify_core()['status']=='PASS'

def test_frozen_schedule():
    p=read(ROOT/'configs/challenge.json');validate(p)
    assert len(development_jobs(p))==8 and len(confirmation_jobs(p))==54
    assert len({(j['dataset'],j['trial_id']) for j in development_jobs(p)+confirmation_jobs(p)})==62
    q=copy.deepcopy(p);q['fit']['difference_weight']=.5
    with pytest.raises(ValueError):validate(q)

def test_missing_data_no_test_release(tmp_path):
    from bc1.experiment import evaluate
    p=read(ROOT/'configs/smoke.json');cfg=p['fit'];cfg=dict(cfg,seeds=[937],methods=p['methods'])
    result=evaluate(cfg,tmp_path)
    assert result['status']=='INCONCLUSIVE' and not (tmp_path/'TEST_RELEASE_LOCK.json').exists()

def test_dev_validation_selection_only(tmp_path):
    p=read(ROOT/'configs/smoke.json')
    for j in development_jobs(p):
        td=tmp_path/'development/trials'/j['trial_id'];td.mkdir(parents=True)
        (td/'best.pt').write_bytes(b'not loaded during LR selection')
        write(td/'parent.json',{'status':'COMPLETE'})
        write(td/'result.json',{'status':'COMPLETE','updates':p['fit']['updates'],'checkpoint_sha256':sha(td/'best.pt'),'validation_score':1. if j['learning_rate']==.001 else 2.})
    rec=select_development(tmp_path,p)
    assert all(rec['learning_rates'][m]==.001 for m in NEW_METHODS)
    assert rec['learning_rates']['fixed_behavior']==.003
    with pytest.raises(RuntimeError):select_development(tmp_path,p)

def test_development_timeout_is_not_no_signal(tmp_path):
    p=read(ROOT/'configs/smoke.json')
    with pytest.raises(RuntimeError):select_development(tmp_path,p)
    assert not (tmp_path/'HYPERPARAMETER_LOCK.json').exists()

def test_cpu_no_cuda_record(tmp_path):
    if torch.cuda.is_available():pytest.skip('No-CUDA-only integration test')
    from bc1.runner import run_stage
    out=tmp_path/'no_gpu';r=run_stage('gpu','cuda:0',out)
    assert r==2 and read(out/'DECISION.json')['status']=='NOT_RUN'
    assert len(read(out/'UNRUN.json'))==62
    assert not (out/'TEST_RELEASE_LOCK.json').exists()

def test_reverse_margin_is_not_winner_picking():
    from bc1.analysis import comparison
    p=read(ROOT/'configs/smoke.json');rows=[]
    for m in p['methods']:
      for case in ('ring4','ring8','long8','reset_iso','reset_ring4'):
        rows.append(dict(method=m,seed=937,case=case,wait=0 if case.startswith(('ring','long')) else 2,mse=1.,difference_error=.4))
    r=comparison(rows,[937],2,'gru32_behavior',p['decision'])
    assert r['baseline_matches_within_5pct'] and not r['candidate_passes_original_gate']
