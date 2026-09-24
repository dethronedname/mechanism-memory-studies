"""Privileged-known-equation positive control, NOT a fair neural baseline.
Only observed calibration u,p enter optimization; no theta labels or query targets.
Numerical quadrature approximates the integrated law; errors are not zero by fiat.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from pi2.simulator import DT,step
from .io import write

def trap_integral(v,dt=DT):
    return np.concatenate([np.zeros(v.shape[:-1]+(1,)),np.cumsum((v[...,:-1]+v[...,1:])*.5*dt,axis=-1)],-1)

def coefficients(io,beta=.2):
    """io: episodes,time,[u,p]. Returns k,d and episode-specific unknown q0s."""
    io=np.asarray(io,dtype='float64')
    if io.ndim!=3 or io.shape[-1]!=2 or io.shape[1]<6:raise ValueError('need E,T,2 observed calibration')
    u,p=io[...,0],io[...,1];e,length=p.shape;t=np.arange(length)*DT
    disp=trap_integral(p);a1=trap_integral(disp);a2=trap_integral(disp**2);a3=trap_integral(disp**3)
    ui=np.concatenate([np.zeros((e,1)),DT*np.cumsum(u[:,:-1],axis=1)],1)
    base=p-p[:,0,None]-ui
    def residual(x):
        k,d=x[:2];q=x[2:,None]
        r=base+k*(q*t+a1)+beta*(q**3*t+3*q*q*a1+3*q*a2+a3)+d*disp
        return r[:,1:].ravel()
    def jac(x):
        k,d=x[:2];q=x[2:,None];j=np.zeros((e,length-1,e+2))
        j[:,:,0]=(q*t+a1)[:,1:];j[:,:,1]=disp[:,1:]
        dq=k*t+beta*(3*q*q*t+6*q*a1+3*a2)
        for i in range(e):j[i,:,i+2]=dq[i,1:]
        return j.reshape(-1,e+2)
    x=np.r_[1.,.2,np.zeros(e)]
    fit=least_squares(residual,x,jac=jac,bounds=(np.r_[.05,0.,np.full(e,-5.)],np.r_[3.,1.,np.full(e,5.)]),
                      max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    return fit.x,{'success':bool(fit.success),'nfev':int(fit.nfev),'residual_mse':float(np.mean(fit.fun**2)),'jacobian_condition':float(np.linalg.cond(fit.jac))}

def predict_reset(support,external,k):
    # B,W,N,E,T,2. No true mechanism information is passed.
    b,w,n,e,t,ch=support.shape;theta=np.empty((b,w,n,2));q0=np.empty((b,w,n,e));fits=[]
    for i in range(b):
        for j in range(w):
            for node in range(n):
                x,r=coefficients(support[i,j,node]);theta[i,j,node]=x[:2];q0[i,j,node]=x[2:];fits.append(r)
    x=np.zeros((b*w,n,2));ys=[]
    for t in range(external.shape[2]):
        x=step(x,external[:,:,t].reshape(b*w,n),k,theta.reshape(b*w,n,2));ys.append(x[...,1].copy())
    y=np.stack(ys,1).reshape(b,w,external.shape[2],n)
    return y,theta,q0,fits

def run(datadir,out):
    datadir=Path(datadir);out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[];started=time.monotonic()
    for case in ['reset_iso','reset_ring4','reset_null']:
        with np.load(datadir/f'{case}.npz') as z:d={k:z[k].copy() for k in z.files}
        # Fit completed BEFORE loading scoring-only theta; targets never passed to fit.
        pa,ta,qa,fa=predict_reset(d['support_a'],d['external'],d['k'])
        pb,tb,qb,fb=predict_reset(d['support_b'],d['external'],d['k'])
        np.savez_compressed(out/f'{case}.npz',prediction=pa,alternate=pb,theta_a=ta,theta_b=tb,q0_a=qa,q0_b=qb)
        with np.load(datadir/'scoring_only'/f'{case}.npz') as z:truth=z['theta']
        dy=d['target'][:,1].astype('float64')-d['target'][:,0];dp=pa[:,1]-pa[:,0];den=float(np.mean(dy**2))
        rows.append({'case':case,'mse':float(np.mean((pa-d['target'])**2)),
                     'difference_error':float(np.mean((dp-dy)**2)/den) if den>1e-15 else None,
                     'difference_mse':float(np.mean((dp-dy)**2)),
                     'same_mechanism_repeat_mse':float(np.mean((pa-pb)**2)),
                     'k_rmse':float(np.sqrt(np.mean((ta[...,0]-truth[...,0])**2))),
                     'd_rmse':float(np.sqrt(np.mean((ta[...,1]-truth[...,1])**2))),
                     'support_k_disagreement_rmse':float(np.sqrt(np.mean((ta[...,0]-tb[...,0])**2))),
                     'support_d_disagreement_rmse':float(np.sqrt(np.mean((ta[...,1]-tb[...,1])**2))),
                     'fits':fa+fb})
    summary={'status':'COMPLETE','prior':'true continuous ODE form, cubic coefficient 0.2, mass 1, exact reset; no parameter labels in fit',
             'comparison':'privileged identifiability positive control, not a same-prior learning competition',
             'seconds':time.monotonic()-started,'records':rows,'wait_behavior':'all waiting lengths identical by exact known reset; not a learned memory result'}
    write(out/'summary.json',summary);return summary
