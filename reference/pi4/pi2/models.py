"""All models see identical support/query I/O; fast state never receives future truth.
PI2 is an adaptation/ablation of existing conditional pH ideas, NOT a novelty claim.
Candidate z2+c2 and primary unpartitioned z4 have equal persistent scalar counts.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .simulator import DT,PROBE_STEPS
FAMILIES=('fixed_context_ph','free_memory_ph','compressed_ph','fixed_context_ode')

class Energy(nn.Module):
    def __init__(self,d,context,width=12):
        super().__init__();self.d=d;self.context=context
        self.factor=nn.Parameter(torch.eye(d));self.a=nn.Parameter(.15*torch.randn(width,d))
        self.b=nn.Parameter(torch.zeros(width));self.w=nn.Parameter(torch.full((width,),-2.))
        if context:
            self.mod=nn.Linear(context,d+width)
            nn.init.normal_(self.mod.weight,0,.05);nn.init.zeros_(self.mod.bias)
    def forward(self,z,c):
        q=self.factor@self.factor.T+.05*torch.eye(self.d,device=z.device,dtype=z.dtype)
        t=torch.tanh(F.linear(z,self.a,self.b));delta=t-torch.tanh(self.b)
        if self.context:
            mod=self.mod(c);diag=F.softplus(mod[...,:self.d])
            w=F.softplus(self.w+mod[...,self.d:])
        else: diag=torch.zeros_like(z);w=F.softplus(self.w)
        e=.5*(z*(z@q)).sum(-1)+.5*(diag*z.square()).sum(-1)+.5*(w*delta.square()).sum(-1)
        g=z@q+diag*z+(w*delta*(1-t.square()))@self.a
        return e,g

class Model(nn.Module):
    def __init__(self,family,history_width=16,width=24):
        super().__init__()
        if family not in FAMILIES: raise ValueError(family)
        self.family=family;self.history_width=history_width;self.width=width
        self.cdim=2 if family.startswith('fixed_context') else 0
        self.d=4 if family=='free_memory_ph' else 2
        self.observer=nn.GRU(2,history_width,batch_first=True)
        self.context_head=nn.Sequential(nn.Linear(history_width,2),nn.Tanh())
        self.initial=nn.Linear(history_width+2,self.d)
        qdim=self.d+self.cdim
        self.probe_head=nn.Sequential(nn.Linear(qdim,16),nn.Tanh(),nn.Linear(16,2))
        if family!='fixed_context_ode':
            self.energy=Energy(self.d,self.cdim)
            self.pairs=self.d*(self.d-1)//2; self.rsize=self.d*(self.d+1)//2
            self.coeffs=nn.Sequential(nn.Linear(qdim,width),nn.Tanh(),nn.Linear(width,self.pairs+self.rsize+self.d))
            inds=torch.tril_indices(self.d,self.d)
            basis=torch.zeros(self.rsize,self.d,self.d)
            basis[torch.arange(self.rsize),inds[0],inds[1]]=1
            skew=[]
            for i in range(self.d):
                for j in range(i+1,self.d):
                    x=torch.zeros(self.d,self.d);x[i,j]=1;x[j,i]=-1;skew.append(x)
            self.register_buffer('lbasis',basis);self.register_buffer('jbasis',torch.stack(skew))
            with torch.no_grad():
                self.coeffs[-1].weight.mul_(.05);self.coeffs[-1].bias.zero_()
                self.coeffs[-1].bias[0]=.5
                for a,(i,j) in enumerate(zip(inds[0],inds[1])):
                    if i==j: self.coeffs[-1].bias[self.pairs+a]=.3
                self.coeffs[-1].bias[-self.d+1]=1.
        else:
            self.flow=nn.Sequential(nn.Linear(qdim+1,width),nn.Tanh(),nn.Linear(width,self.d))
            self.output=nn.Linear(qdim,1,bias=False)
    def config(self): return dict(family=self.family,history_width=self.history_width,width=self.width)
    @property
    def persistent_dimension(self): return self.d+self.cdim
    def encode_support(self,support):
        b,n,e,t,_=support.shape
        _,h=self.observer(support.reshape(b*n*e,t,2))
        # permutation-invariant pooling over RESET calibration episodes, not trajectory stitching
        return self.context_head(h[-1].reshape(b,n,e,-1).mean(2))
    def initialize(self,history,support):
        b,t,n,_=history.shape
        _,h=self.observer(history.permute(0,2,1,3).reshape(b*n,t,2))
        c=self.encode_support(support)
        z=self.initial(torch.cat((h[-1].reshape(b,n,-1),c),-1))
        # Unpartitioned controls receive exactly the same c in initialization, then discard it.
        return z,c if self.cdim else z.new_empty(b,n,0)
    def joined(self,z,c): return torch.cat((z,c),-1) if self.cdim else z
    def terms(self,z,c):
        e,g=self.energy(z,c);p=self.coeffs(self.joined(z,c))
        j=torch.einsum('...a,aij->...ij',p[...,:self.pairs],self.jbasis)
        l=torch.einsum('...a,aij->...ij',p[...,self.pairs:self.pairs+self.rsize],self.lbasis)
        r=l@l.transpose(-1,-2) # PSD with exact zero directions allowed
        b=p[...,-self.d:]
        return e,g,j,r,b
    def port(self,z,c):
        if self.family=='fixed_context_ode': return self.output(self.joined(z,c)).squeeze(-1)
        _,g=self.energy(z,c);p=self.coeffs(self.joined(z,c))
        return (g*p[...,-self.d:]).sum(-1)
    def derivative(self,z,c,external,k):
        if self.family=='fixed_context_ode':
            y=self.port(z,c);u=external-y@k.T
            return self.flow(torch.cat((self.joined(z,c),u.unsqueeze(-1)),-1))
        _,g,j,r,b=self.terms(z,c);y=(g*b).sum(-1);u=external-y@k.T
        return ((j-r)@g.unsqueeze(-1)).squeeze(-1)+b*u.unsqueeze(-1)
    def advance(self,z,c,external,k,dt=DT):
        a=self.derivative(z,c,external,k)
        return z+.5*dt*(a+self.derivative(z+dt*a,c,external,k))
    def rollout(self,history,support,external,k):
        z,c=self.initialize(history,support);states=[z];ys=[]
        for t in range(external.shape[1]):
            z=self.advance(z,c,external[:,t],k);states.append(z);ys.append(self.port(z,c))
        return torch.stack(ys,1),torch.stack(states,1),c
    def probe(self,z,c):
        b,n,_=z.shape;z=z.reshape(b*n,1,self.d);c=c.reshape(b*n,1,self.cdim)
        k=z.new_zeros(1,1);u=z.new_zeros(b*n,1);ys=[]
        for t in range(1,max(PROBE_STEPS)+1):
            z=self.advance(z,c,u,k)
            if t in PROBE_STEPS: ys.append(self.port(z,c).reshape(b,n))
        return torch.stack(ys,-1)
    def objective(self,batch):
        yp,z,c=self.rollout(batch['history'],batch['support'],batch['external'],batch['k'])
        mse=(yp-batch['target']).square().mean()
        ini=(self.port(z[:,0],c)-batch['y0']).square().mean()
        sel=z[:,batch['snapshot']];b,j,n,d=sel.shape
        zz=sel.reshape(b*j,n,d);cc=c[:,None].expand(b,j,n,self.cdim).reshape(b*j,n,self.cdim)
        branch=self.probe(zz,cc).reshape(b,j,n,2)
        aux=self.probe_head(self.joined(zz,cc)).reshape(b,j,n,2)
        bl=(branch-batch['probe']).square().mean();al=(aux-batch['probe']).square().mean()
        return mse+.25*ini+.5*bl+.5*al,{'mse':float(mse.detach()),'initial':float(ini.detach()),'branch':float(bl.detach()),'aux':float(al.detach())}
