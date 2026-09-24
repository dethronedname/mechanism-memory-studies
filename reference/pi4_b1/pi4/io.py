from __future__ import annotations
import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    os.replace(temp,path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def array_hash(d):
    h=hashlib.sha256()
    for key in sorted(d):
        a=np.ascontiguousarray(d[key]);h.update(key.encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()

def load(path,device='cpu'):
    with np.load(path,allow_pickle=False) as z:return {k:torch.from_numpy(z[k].copy()).to(device) for k in z.files}

def take(data,ix):
    return {k:v if k in ('k','snapshot') else v[ix] for k,v in data.items()}
