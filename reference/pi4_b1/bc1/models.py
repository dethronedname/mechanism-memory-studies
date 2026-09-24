"""Task-adapted conventional baselines, NOT official-paper reproductions.

All new baselines reread a fixed calibration embedding on EVERY update and output.
The encoder follows the old task's episode pooling. Larger models deliberately
receive more persistent memory; comparison is not parameter/memory matching.
"""
from __future__ import annotations
import torch
from torch import nn
from pi2.models import Model as LegacyModel
from pi2.simulator import DT, PROBE_STEPS

METHODS={
 'fixed_behavior':('fixed_context_ph',True),
 'free_behavior':('free_memory_ph',True),
 'gru4_behavior':('calibration_gru_d2_c2',True),
 'gru32_behavior':('calibration_gru_d16_c16',True),
 'graph32_behavior':('calibration_graph_gru_d16_c16',True),
 'node32_behavior':('calibration_node_d16_c16',True),
}
NEW_METHODS=tuple(list(METHODS)[2:])

class ContextDynamics(nn.Module):
    def __init__(self,method:str,history_width:int=16,width:int=24):
        super().__init__()
        if method not in NEW_METHODS:raise ValueError(method)
        self.method=method;self.family=METHODS[method][0]
        self.history_width=history_width;self.width=width
        self.d=self.cdim=2 if method=='gru4_behavior' else 16
        self.hidden=24 if self.d==2 else 32
        self.observer=nn.GRU(2,history_width,batch_first=True)
        self.context_head=nn.Sequential(nn.Linear(history_width,self.cdim),nn.Tanh())
        self.initial=nn.Linear(history_width+self.cdim,self.d)
        self.probe_head=nn.Sequential(nn.Linear(self.d+self.cdim,16),nn.Tanh(),nn.Linear(16,2))
        self.output=nn.Linear(self.d+self.cdim,1,bias=False)
        if method=='node32_behavior':
            self.flow=nn.Sequential(nn.Linear(self.d+self.cdim+1,self.hidden),nn.Tanh(),nn.Linear(self.hidden,self.d))
            # Common small last-layer initialization, fixed before any fitting.
            with torch.no_grad():self.flow[-1].weight.mul_(.1);self.flow[-1].bias.zero_()
        else:
            size=self.cdim+2
            if method=='graph32_behavior':
                self.message=nn.Sequential(nn.Linear(2*(self.d+self.cdim)+1,self.hidden),nn.SiLU(),nn.Linear(self.hidden,self.hidden))
                size+=self.hidden
            self.cell=nn.GRUCell(size,self.d)
            # No retention gate hand-setting or equilibrium/identity-state oracle.

    def config(self):return dict(method=self.method,history_width=self.history_width,width=self.width)
    @property
    def persistent_dimension(self):return self.d+self.cdim
    def joined(self,z,c):return torch.cat((z,c),-1)
    def encode_support(self,support):
        b,n,e,t,ch=support.shape
        if ch!=2:raise ValueError('Expected observed input/output only')
        _,h=self.observer(support.reshape(b*n*e,t,2))
        return self.context_head(h[-1].reshape(b,n,e,-1).mean(2))
    def initialize(self,history,support):
        b,t,n,ch=history.shape
        _,h=self.observer(history.permute(0,2,1,3).reshape(b*n,t,ch))
        c=self.encode_support(support)
        z=self.initial(torch.cat((h[-1].reshape(b,n,-1),c),-1))
        return z,c
    def port(self,z,c):return self.output(self.joined(z,c)).squeeze(-1)
    def _messages(self,z,c,k):
        b,n,_=z.shape;x=self.joined(z,c)
        recv=x[:,:,None,:].expand(-1,-1,n,-1)
        send=x[:,None,:,:].expand(-1,n,-1,-1)
        edge=k[None,:,:,None].expand(b,-1,-1,-1)
        msg=self.message(torch.cat((recv,send,edge),-1))
        # Sum aggregation is permutation equivariant; |K| masks absent/self edges.
        # Signed K is also provided. This is one message round, not GNS reproduction.
        return (msg*k.abs()[None,:,:,None]).sum(2)
    def derivative(self,z,c,external,k):
        if self.method!='node32_behavior':raise TypeError('GRU is a discrete step model')
        y=self.port(z,c);u=external-y@k.T
        return self.flow(torch.cat((z,c,u.unsqueeze(-1)),-1))
    def advance(self,z,c,external,k,dt=DT):
        if self.method=='node32_behavior':
            a=self.derivative(z,c,external,k)
            return z+.5*dt*(a+self.derivative(z+dt*a,c,external,k))
        if abs(float(dt)-DT)>1e-12:raise ValueError('GRU baseline trained for the fixed sampling interval only')
        y=self.port(z,c);u=external-y@k.T
        pieces=[c,u.unsqueeze(-1),y.unsqueeze(-1)]
        if self.method=='graph32_behavior':pieces.append(self._messages(z,c,k))
        inp=torch.cat(pieces,-1);shape=z.shape
        return self.cell(inp.reshape(-1,inp.shape[-1]),z.reshape(-1,self.d)).reshape(shape)
    def probe(self,z,c):
        b,n,_=z.shape;z=z.reshape(b*n,1,self.d);c=c.reshape(b*n,1,self.cdim)
        k=z.new_zeros(1,1);u=z.new_zeros(b*n,1);ys=[]
        for j in range(1,max(PROBE_STEPS)+1):
            z=self.advance(z,c,u,k)
            if j in PROBE_STEPS:ys.append(self.port(z,c).reshape(b,n))
        return torch.stack(ys,-1)

def model_for(method,width=24,history_width=16):
    if method not in METHODS:raise ValueError(method)
    if method in ('fixed_behavior','free_behavior'):
        return LegacyModel(METHODS[method][0],history_width,width)
    return ContextDynamics(method,history_width,width)
