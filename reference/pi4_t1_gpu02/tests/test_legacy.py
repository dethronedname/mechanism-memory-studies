from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from pi2.models import Model
from pi2.simulator import graph,step
from pi4.data import general,reset_pairs,make
from pi4.learning import METHODS,model_for,general_views,reset_views,terms_reset,objective
from pi4.io import array_hash,load,take

torch.set_num_threads(1)
CONFIG=json.loads((Path(__file__).parents[1]/'configs/cpu.json').read_text())
def tensors(d):return {k:torch.from_numpy(v) for k,v in d.items()}

@pytest.mark.parametrize('kind',['general','reset'])
def test_reproducible(kind):
 f=general if kind=='general' else reset_pairs
 x,t=f(3,2,'chain',8,6,800,episodes=2,length=12);xx,tt=f(3,2,'chain',8,6,800,episodes=2,length=12)
 assert array_hash(x)==array_hash(xx);np.testing.assert_array_equal(t,tt)
 assert not any(k in x for k in ['theta','q','state','stiffness','damping'])

def test_reset_matching_and_null():
 d,t=reset_pairs(4,1,'isolated',16,8,81,length=12)
 np.testing.assert_array_equal(d['external'][:,0],d['external'][:,1]);assert not np.array_equal(d['target'][:,0],d['target'][:,1])
 np.testing.assert_array_equal(d['support_a'][:,0,...,0],d['support_a'][:,1,...,0])
 np.testing.assert_array_equal(d['support_a'][:,0,:,:,0,1],d['support_a'][:,1,:,:,0,1])
 assert not np.array_equal(d['support_a'],d['support_b']);assert (d['history']==0).all()
 nd,nt=reset_pairs(4,1,'isolated',16,8,81,length=12,null=True)
 np.testing.assert_array_equal(nd['target'][:,0],nd['target'][:,1]);np.testing.assert_array_equal(nd['support_a'][:,0],nd['support_a'][:,1])

def test_pair_loss_identity():
 torch.manual_seed(77);p=torch.randn(6,2,2,7,3,dtype=torch.double);y=torch.randn(6,2,7,3,dtype=torch.double)
 mse,rep,diff=terms_reset(p,y)
 common=(p.mean((1,2))-y.mean(1)).square().mean()
 torch.testing.assert_close(mse,common+.25*diff+.25*rep,atol=1e-14,rtol=1e-14)

def test_repeat_alone_allows_collapse():
 p=torch.zeros(3,2,2,4,1);y=torch.ones(3,2,4,1);y[:,0]*=-1
 mse,r,diff=terms_reset(p,y);assert r==0 and diff>0 and mse>0

@pytest.mark.parametrize('pair',[('fixed_mse','fixed_behavior'),('free_mse','free_behavior')])
def test_identical_initialization(pair):
 torch.manual_seed(201);a=model_for(pair[0]);torch.manual_seed(201);b=model_for(pair[1])
 for k in a.state_dict():torch.testing.assert_close(a.state_dict()[k],b.state_dict()[k],atol=0,rtol=0)
 assert a.persistent_dimension==4 and b.persistent_dimension==4

@pytest.mark.parametrize('method',list(METHODS))
@pytest.mark.parametrize('kind',['general','reset'])
def test_complete_gradient_path(method,kind):
 torch.manual_seed(44);model=model_for(method)
 data,_=(general if kind=='general' else reset_pairs)(2,1,'isolated',6,5,80,length=8)
 d=tensors(data);scales={'general':1.,'reset':1.,'difference':.2}
 loss,parts=objective(model,d,kind,METHODS[method][1],scales,CONFIG,wait=2 if kind=='reset' else 0)
 loss.backward();assert torch.isfinite(loss)
 grads=[p.grad for p in model.parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(g).all() for g in grads)
 assert sum(float(g.abs().sum()) for g in grads)>0
 assert model.context_head[0].weight.grad.abs().sum()>0

@pytest.mark.parametrize('method',['fixed_behavior','free_behavior','ode_behavior'])
def test_no_future_target_leak_and_support_permutation(method):
 torch.manual_seed(98);m=model_for(method);d,_=reset_pairs(2,1,'isolated',8,6,74,length=9);d=tensors(d)
 with torch.no_grad():p,_,_=reset_views(m,d)
 d['target'][:]=12345
 with torch.no_grad():q,_,_=reset_views(m,d)
 torch.testing.assert_close(p,q,atol=0,rtol=0)
 d['support_a']=d['support_a'].flip(3);d['support_b']=d['support_b'].flip(3)
 with torch.no_grad():q,_,_=reset_views(m,d)
 torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-6)

def test_wait_true_system_equilibrium():
 theta=np.array([[[1.2,.18]],[[.6,.3]]]);x=np.zeros((2,1,2));k=graph(1,'isolated')
 for i in range(256):x=step(x,np.zeros((2,1)),k,theta)
 assert (x==0).all()

def test_same_model_node_permutation():
 d,_=general(2,4,'ring',8,6,80,length=8);d=tensors(d);torch.manual_seed(20);m=model_for('fixed_behavior')
 with torch.no_grad():p,_,_=general_views(m,d)
 perm=torch.tensor([2,0,3,1]);q={**d}
 for name in ['support_a','support_b']:q[name]=d[name][:,perm]
 q['history']=d['history'][:,:,perm];q['external']=d['external'][:,:,perm];q['k']=d['k'][perm][:,perm]
 with torch.no_grad():pp,_,_=general_views(m,q)
 torch.testing.assert_close(pp,p[:,:,perm],atol=3e-6,rtol=3e-6)

def test_behavior_zero_weights_matches_mse():
 cfg=dict(CONFIG,difference_weight=0.,consistency_weight=0.)
 d,_=reset_pairs(2,1,'isolated',8,6,54,length=9);d=tensors(d);sc={'general':1.,'reset':1.,'difference':.2}
 torch.manual_seed(10);m=model_for('fixed_behavior');a,_=objective(m,d,'reset',False,sc,cfg)
 b,_=objective(m,d,'reset',True,sc,cfg);torch.testing.assert_close(a,b,atol=0,rtol=0)

def test_generation_scales_train_only(tmp_path):
 cfg=dict(CONFIG,train_count=3,val_count=2,test_count=2,horizon=4,history=4,reset_horizon=6,support_length=8)
 manifest,sc=make(cfg,tmp_path/'data');d=load(tmp_path/'data/train_reset.npz')
 v=float(d['target'].double().square().mean());dd=float((d['target'][:,1].double()-d['target'][:,0].double()).square().mean())
 assert sc['reset']==pytest.approx(v);assert sc['difference']==pytest.approx(dd)
 assert 'theta' not in d and len(manifest)==12

def test_missing_trial_does_not_release_test(tmp_path):
 from pi4.experiment import evaluate
 result=evaluate(CONFIG,tmp_path)
 assert result['status']=='INCONCLUSIVE';assert not (tmp_path/'TEST_RELEASE_LOCK.json').exists()

def test_fixed_context_remains_exact():
 d,_=reset_pairs(2,1,'isolated',8,6,54,length=9);d=tensors(d);m=model_for('fixed_behavior')
 s=d['support_a'][:,0];h=d['history'][:,0];z,c=m.initialize(h,s);old=c.clone()
 for i in range(16):z=m.advance(z,c,z.new_zeros(2,1),z.new_zeros(1,1))
 torch.testing.assert_close(c,old,atol=0,rtol=0)

def test_free_can_represent_nondissipating_directions():
 # Algebraic counterexample: PSD does NOT require identity information to decay.
 j=torch.zeros(4,4,dtype=torch.double);j[0,1]=1;j[1,0]=-1
 r=torch.diag(torch.tensor([0.,.2,0.,0.],dtype=torch.double));a=j-r
 assert torch.equal(a[2:],torch.zeros(2,4,dtype=torch.double))
 assert int((torch.linalg.eigvals(a).abs()==0).sum())==2

def test_integral_identity_exact_quadrature():
    # Analytic q(t)=q0 + v*t, external follows the correct ODE; integral balance.
    from pi4.greybox import trap_integral
    t=np.linspace(0,2,10001);dt=t[1]-t[0];q0=.4;v=.3;k=.8;d=.2
    q=q0+v*t;u=k*q+.2*q**3+d*v
    bal=-trap_integral(u,dt)+k*(q0*t+.5*v*t*t)+.2*(q0**3*t+1.5*q0*q0*v*t*t+q0*v*v*t**3+.25*v**3*t**4)+d*v*t
    assert np.max(np.abs(bal))<1e-9

def test_greybox_uses_io_and_recovers_mechanism():
    from pi4.greybox import coefficients
    d,t=reset_pairs(1,1,'isolated',6,6,91,episodes=2,length=48)
    x,r=coefficients(d['support_a'][0,0,0]);assert r['success']
    np.testing.assert_allclose(x[:2],t[0,0,0],atol=.015,rtol=0)

def test_model_reload(tmp_path):
    torch.manual_seed(731);m=model_for('fixed_behavior');d,_=reset_pairs(2,1,'isolated',8,5,96,length=8);d=tensors(d)
    with torch.no_grad():p,_,_=reset_views(m,d)
    torch.save({'config':m.config(),'state_dict':m.state_dict()},tmp_path/'model.pt')
    s=torch.load(tmp_path/'model.pt',weights_only=True);m2=Model(**s['config']);m2.load_state_dict(s['state_dict'])
    with torch.no_grad():q,_,_=reset_views(m2,d)
    torch.testing.assert_close(p,q,atol=0,rtol=0)

def test_end_to_end_fit_one_update(tmp_path):
    from pi4.experiment import fit
    cfg=dict(CONFIG,train_count=4,val_count=2,test_count=2,horizon=4,history=4,reset_horizon=6,support_length=8,
             general_batch=2,pair_batch=2,updates=2,validate_every=1)
    make(cfg,tmp_path/'data')
    r=fit(cfg,'fixed_behavior',501,tmp_path/'data',tmp_path/'trial','cpu')
    assert r['status']=='COMPLETE' and r['updates']==2 and (tmp_path/'trial/best.pt').exists()
    curve=json.loads((tmp_path/'trial/curve.json').read_text())
    assert r['validation_score']==min(x['selection_score'] for x in curve)

def test_unexcited_calibration_cannot_identify():
    th=np.array([[[.6,.1]],[[1.4,.3]]]);x=np.zeros((2,1,2));k=np.zeros((1,1))
    for _ in range(20):x=step(x,np.zeros((2,1)),k,th)
    np.testing.assert_array_equal(x[0],x[1])
    for _ in range(20):x=step(x,np.ones((2,1))*.5,k,th)
    assert abs(x[0,0,1]-x[1,0,1])>.01
