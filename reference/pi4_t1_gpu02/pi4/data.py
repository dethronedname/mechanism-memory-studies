"""Paired observed trajectories. No theta or hidden state in learner-facing files.
A and B denote independent calibration nuisances for the SAME mechanism.
The two worlds in a pair have matched nuisance/input but distinct mechanisms.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from pi2.simulator import DT,graph,step,drive,episode,probes
from .io import write,array_hash

def theta_sample(rng,count,n):
    return np.stack([rng.uniform(.5,1.5,(count,n)),rng.uniform(.05,.35,(count,n))],-1)

def cal_single(rng,theta,episodes,length):
    count,n,_=theta.shape;out=[]
    for _ in range(episodes):
        _,_,io=episode(rng,count,n,length,np.zeros((n,n)),theta)
        out.append(io.transpose(0,2,1,3))
    return np.stack(out,2)

def cal_pair(rng,theta,episodes,length):
    count,two,n,_=theta.shape
    if two!=2:raise ValueError('exactly two paired worlds required')
    out=[]
    for _ in range(episodes):
        init=rng.normal(0,.7,(count,n,2));u=drive(rng,count,n,length)
        z=np.repeat(init[:,None],2,1).reshape(count*2,n,2)
        ext=np.repeat(u[:,None],2,1).reshape(count*2,length,n)
        ys=[z[...,1].copy()]
        for t in range(length-1):
            z=step(z,ext[:,t],np.zeros((n,n)),theta.reshape(count*2,n,2));ys.append(z[...,1].copy())
        y=np.stack(ys,1).reshape(count,2,length,n)
        io=np.stack([np.repeat(u[:,None],2,1),y],-1).transpose(0,1,3,2,4)
        out.append(io)
    return np.stack(out,3) # pair, world, node, episode, time, channel

def general(count,n,kind,horizon,history,seed,episodes=2,length=48):
    rng=np.random.default_rng(seed);theta=theta_sample(rng,count,n);k=graph(n,kind)
    a=cal_single(rng,theta,episodes,length);b=cal_single(rng,theta,episodes,length)
    states,ext,io=episode(rng,count,n,history+horizon,k,theta)
    start=history-1;snap=np.array([0,horizon//2,horizon],dtype='int64')
    data={'support_a':a,'support_b':b,'history':io[:,:history],
          'external':ext[:,start:start+horizon],'target':states[:,start+1:start+horizon+1,:,1],
          'k':k,'y0':states[:,start,:,1],
          'probe':np.stack([probes(states[:,start+j],theta) for j in snap],1),'snapshot':snap}
    return {key:v.astype('float32') if key!='snapshot' else v for key,v in data.items()},theta

def reset_pairs(count,n,kind,horizon,history,seed,episodes=2,length=48,null=False):
    rng=np.random.default_rng(seed);t0=theta_sample(rng,count,n);t1=theta_sample(rng,count,n)
    if null:t1=t0.copy()
    theta=np.stack([t0,t1],1);k=graph(n,kind)
    a=cal_pair(rng,theta,episodes,length);b=cal_pair(rng,theta,episodes,length)
    # Exact zero reset is an observable experimental condition for ALL learners.
    ext=drive(rng,count,n,horizon);ext=np.repeat(ext[:,None],2,1)
    z=np.zeros((count*2,n,2));ys=[]
    for t in range(horizon):
        z=step(z,ext[:,:,t].reshape(count*2,n),k,theta.reshape(count*2,n,2));ys.append(z[...,1].copy())
    target=np.stack(ys,1).reshape(count,2,horizon,n)
    data={'support_a':a,'support_b':b,'history':np.zeros((count,2,history,n,2)),
          'external':ext,'target':target,'k':k,'y0':np.zeros((count,2,n))}
    return {key:v.astype('float32') for key,v in data.items()},theta

def make(config,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'scoring_only').mkdir();manifest={};base=config['data_seed']
    specs=[]
    for phase,offset in [('train',0),('val',100)]:
        for n in (2,4):
            specs.append((f'{phase}_g{n}','general',config[phase+'_count'],n,'chain',config['horizon'],base+offset+n,False))
        specs.append((f'{phase}_reset','reset',config[phase+'_count'],1,'isolated',config['reset_horizon'],base+offset+11,False))
    for i,(name,n,kind,h,mode,null) in enumerate([
        ('ring4',4,'ring',config['horizon'],'general',False),
        ('ring8',8,'ring',config['horizon'],'general',False),
        ('long8',8,'ring',4*config['horizon'],'general',False),
        ('reset_iso',1,'isolated',64,'reset',False),
        ('reset_ring4',4,'ring',64,'reset',False),
        ('reset_null',1,'isolated',64,'reset',True)]):
        specs.append((name,mode,config['test_count'],n,kind,h,base+200+i,null))
    for name,mode,count,n,kind,h,seed,null in specs:
        kw=dict(count=count,n=n,kind=kind,horizon=h,history=config['history'],seed=seed,
                episodes=config['support_count'],length=config['support_length'])
        data,theta=general(**kw) if mode=='general' else reset_pairs(**kw,null=null)
        np.savez_compressed(out/f'{name}.npz',**data)
        np.savez_compressed(out/'scoring_only'/f'{name}.npz',theta=theta)
        manifest[name]=dict(mode=mode,count=count,n=n,horizon=h,seed=seed,null=null,array_hash=array_hash(data))
    # Fixed normalizers use TRAINING labels only; no per-test/pair divisions in objective.
    with np.load(out/'train_reset.npz') as z:
        y=z['target'].astype('float64');v=float(np.mean(y*y));diff=float(np.mean((y[:,1]-y[:,0])**2))
    gen_num=0.;gen_n=0
    for n in (2,4):
        with np.load(out/f'train_g{n}.npz') as z:gen_num+=float(np.sum(z['target'].astype('float64')**2));gen_n+=z['target'].size
    scales={'general':max(gen_num/gen_n,1e-6),'reset':max(v,1e-6),'difference':max(diff,1e-6)}
    write(out/'MANIFEST.json',manifest);write(out/'SCALES.json',scales);return manifest,scales
