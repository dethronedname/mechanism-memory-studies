"""Private heterogeneous oscillator simulator. Learners get I/O only.
Calibration episodes reset fast state but preserve per-component (stiffness,damping).
Different components/episodes have different parameters; topology is known and skew.
No coefficient/state labels are included in the learner-facing return dictionary.
"""
from __future__ import annotations
import numpy as np
DT=0.1
PROBE_STEPS=(1,4)

def graph(n:int,kind:str,strength:float=.35)->np.ndarray:
    if n<1: raise ValueError('n must be positive')
    k=np.zeros((n,n),dtype=np.float64)
    if kind=='isolated': return k
    if kind=='chain': edges=[(i,i+1) for i in range(n-1)]
    elif kind=='ring' and n>=3: edges=[(i,(i+1)%n) for i in range(n)]
    elif kind=='star': edges=[(0,i) for i in range(1,n)]
    else: raise ValueError(kind)
    for i,j in edges: k[i,j]+=strength;k[j,i]-=strength
    return k

def rhs(x,external,k,theta):
    q,p=x[...,0],x[...,1]
    net=external-p@k.T
    return np.stack((p,-theta[...,0]*q-.2*q**3-theta[...,1]*p+net),-1)

def step(x,external,k,theta,dt=DT,substeps=4):
    z=x.copy();h=dt/substeps
    for _ in range(substeps):
        a=rhs(z,external,k,theta);b=rhs(z+h*a/2,external,k,theta)
        c=rhs(z+h*b/2,external,k,theta);d=rhs(z+h*c,external,k,theta)
        z=z+h*(a+2*b+2*c+d)/6
    return z

def energy(x,theta):
    return .5*x[...,1]**2+.5*theta[...,0]*x[...,0]**2+.05*x[...,0]**4

def drive(rng,count,n,length):
    t=np.arange(length)*DT
    amp=rng.uniform(.15,.8,(count,n,3));freq=rng.uniform(.35,2.,(count,n,3))
    phase=rng.uniform(-np.pi,np.pi,(count,n,3))
    return (amp[...,None]*np.sin(freq[...,None]*t+phase[...,None])).sum(2).transpose(0,2,1)

def episode(rng,count,n,length,k,theta):
    x=rng.normal(0,.7,(count,n,2));ext=drive(rng,count,n,length)
    states=[x.copy()]
    for t in range(length-1): x=step(x,ext[:,t],k,theta);states.append(x.copy())
    states=np.stack(states,1);y=states[...,1];u=ext-y@k.T
    return states,ext,np.stack((u,y),-1)

def probes(x,theta):
    b,n,_=x.shape; z=x.reshape(b*n,1,2).copy();th=theta.reshape(b*n,1,2)
    k=np.zeros((1,1));u=np.zeros((b*n,1));out=[]
    for t in range(1,max(PROBE_STEPS)+1):
        z=step(z,u,k,th)
        if t in PROBE_STEPS: out.append(z[...,1].reshape(b,n))
    return np.stack(out,-1)

def sample(count,n,kind,horizon,history,seed,support_count=2,support_length=48,homogeneous=False):
    if min(count,horizon,history,support_count,support_length)<1: raise ValueError('invalid sizes')
    rng=np.random.default_rng(seed);k=graph(n,kind)
    theta=np.stack((rng.uniform(.5,1.5,(count,n)),rng.uniform(.05,.35,(count,n))),-1)
    if homogeneous: theta[...]=np.array([1.,.2])
    # Calibration is isolated and state-reset, neither a contiguous query prefix nor a forecast.
    support=[]
    for _ in range(support_count):
        _,_,io=episode(rng,count,n,support_length,np.zeros((n,n)),theta)
        support.append(io.transpose(0,2,1,3))
    support=np.stack(support,2)  # B,N,E,T,2
    states,ext,io=episode(rng,count,n,history+horizon,k,theta)
    start=history-1;snap=np.array([0,horizon//2,horizon],dtype=np.int64)
    probe=np.stack([probes(states[:,start+j],theta) for j in snap],1)
    return {'support':support.astype('float32'),'history':io[:,:history].astype('float32'),
            'external':ext[:,start:start+horizon].astype('float32'),'k':k.astype('float32'),
            'target':states[:,start+1:start+horizon+1,:,1].astype('float32'),
            'y0':states[:,start,:,1].astype('float32'),'probe':probe.astype('float32'),'snapshot':snap}
