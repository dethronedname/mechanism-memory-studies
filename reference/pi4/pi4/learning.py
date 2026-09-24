"""Same five old inference graphs, two objective variants; NO extra mechanism head.
The 2x2 primary factorial distinguishes fixed state from behavioral supervision.
"""
from __future__ import annotations
import torch
from pi2.models import Model

METHODS={
 'fixed_mse':('fixed_context_ph',False),
 'fixed_behavior':('fixed_context_ph',True),
 'free_mse':('free_memory_ph',False),
 'free_behavior':('free_memory_ph',True),
 'ode_behavior':('fixed_context_ode',True),
}

def model_for(method,width=24,history_width=16):
    return Model(METHODS[method][0],history_width,width)

def rollout(model,history,support,external,k,wait=0,states=False):
    if wait<0:raise ValueError('negative wait')
    z,c=model.initialize(history,support);zs=[z];ys=[]
    for _ in range(wait):z=model.advance(z,c,z.new_zeros(z.shape[:-1]),k)
    if wait:zs=[z]
    for t in range(external.shape[1]):
        z=model.advance(z,c,external[:,t],k);ys.append(model.port(z,c));
        if states:zs.append(z)
    return torch.stack(ys,1),torch.stack(zs,1),c

def general_views(model,d,states=False,wrong=False):
    a,b=d['support_a'],d['support_b'];a=torch.roll(a,1,dims=0) if wrong else a
    hist=torch.cat([d['history'],d['history']],0);support=torch.cat([a,b],0)
    ext=torch.cat([d['external'],d['external']],0)
    return rollout(model,hist,support,ext,d['k'],states=states)

def reset_views(model,d,wait=0,states=False,wrong=False):
    b,two,h,n,c=d['history'].shape
    def fl(x):return x.reshape(b*two,*x.shape[2:])
    a=d['support_a']
    if wrong:a=a.flip(1) # matched-nuisance, wrong mechanism; never used in training
    hist=torch.cat([fl(d['history'])]*2,0)
    ext=torch.cat([fl(d['external'])]*2,0)
    support=torch.cat([fl(a),fl(d['support_b'])],0)
    yp,z,c=rollout(model,hist,support,ext,d['k'],wait=wait,states=states)
    # 2 views, B pairs, 2 worlds -> B, world, view, time, node
    return yp.reshape(2,b,2,*yp.shape[1:]).permute(1,2,0,3,4),z,c

def terms_reset(pred,target):
    err=pred-target[:,:,None]
    mse=err.square().mean()
    repeated=(pred[:,:,0]-pred[:,:,1]).square().mean()
    delta=pred.mean(2)[:,1]-pred.mean(2)[:,0]
    target_delta=target[:,1]-target[:,0]
    diff=(delta-target_delta).square().mean()
    return mse,repeated,diff

def objective(model,d,kind,behavior,scales,config,wait=0):
    if kind=='general':
        p,z,c=general_views(model,d,states=True);b=d['history'].shape[0]
        y=torch.cat([d['target']]*2,0);mse=(p-y).square().mean()
        ini=(model.port(z[:,0],c)-torch.cat([d['y0']]*2,0)).square().mean()
        selected=z[:,d['snapshot']];bb,j,n,dim=selected.shape
        zz=selected.reshape(bb*j,n,dim);cc=c[:,None].expand(bb,j,n,model.cdim).reshape(bb*j,n,model.cdim)
        branch=model.probe(zz,cc).reshape(bb,j,n,2)
        aux=model.probe_head(model.joined(zz,cc)).reshape(bb,j,n,2)
        probes=torch.cat([d['probe']]*2,0)
        bl=(branch-probes).square().mean();al=(aux-probes).square().mean()
        repeat=(p[:b]-p[b:]).square().mean()
        loss=(mse+.25*ini+.5*bl+.5*al)/scales['general']
        if behavior:loss=loss+config['consistency_weight']*repeat/scales['general']
        parts={'mse':mse,'repeat':repeat,'initial':ini,'probe':bl,'aux':al}
    else:
        p,z,c=reset_views(model,d,wait=wait,states=False)
        mse,repeat,diff=terms_reset(p,d['target'])
        initial=model.port(z[:,0],c).square().mean() # after wait, true output is still zero
        loss=(mse+.25*initial)/scales['reset']
        if behavior:
            loss=loss+config['difference_weight']*diff/scales['difference']+config['consistency_weight']*repeat/scales['reset']
        parts={'mse':mse,'repeat':repeat,'difference':diff,'initial':initial}
    return loss,{k:float(v.detach()) for k,v in parts.items()}
