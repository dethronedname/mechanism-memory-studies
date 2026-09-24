#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys,os,json,time,subprocess,traceback,platform,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pi4.io import write,sha

def worker(args):
    started=time.monotonic() if args.started is None else args.started
    import torch
    from pi4.experiment import fit
    c=json.loads(Path(args.config).read_text())
    try:
        fit(c,args.method,args.seed,args.data,args.out,args.device,started)
        print('COMPLETE',flush=True)
    except Exception as e:
        write(Path(args.out)/'error.json',{'status':'ERROR','error':repr(e),'traceback':traceback.format_exc()});raise

def run(args):
    import torch,numpy,scipy
    from pi4.data import make
    from pi4.experiment import evaluate
    start=time.monotonic();out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('new run directory required: '+str(out))
    out.mkdir(parents=True)
    c=json.loads((ROOT/'configs'/f'{args.stage}.json').read_text());write(out/'config.json',c)
    env={'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
         'requested_device':args.device,'cuda_available':torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES')}
    write(out/'environment.json',env)
    if args.stage=='gpu' and (not args.device.startswith('cuda') or not torch.cuda.is_available()):
        write(out/'summary.json',{'status':'NOT_RUN','gpu':'NOT_RUN','reason':'CUDA required; no CPU fallback'});return 2
    if args.stage=='cpu' and args.device!='cpu':raise ValueError('cpu stage must use CPU')
    frozen={str(p.relative_to(ROOT)):sha(p) for folder in ('pi2','pi4','configs','scripts') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    write(out/'SOURCE_MANIFEST.json',frozen)
    make(c,out/'data');torch.set_num_threads(1)
    if args.stage=='gpu':
        from pi4.preflight import check
        check(c,out/'data',args.device,out/'preflight.json')
    jobs=[(m,s) for s in c['seeds'] for m in c['methods']]
    # Fixed order alternating objective and family per seed; no winner-dependent scheduling.
    for method,seed in jobs:
        if time.monotonic()-start+c['process_limit_seconds']>c['total_limit_seconds']:
            write(out/'RESOURCE_LIMIT.json',{'status':'PARTIAL','elapsed':time.monotonic()-start});break
        d=out/'trials'/f'{method}_s{seed}';d.mkdir(parents=True)
        t=time.monotonic();cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--config',str(out/'config.json'),'--data',str(out/'data'),
                              '--method',method,'--seed',str(seed),'--out',str(d),'--device',args.device,'--started',str(t)]
        with open(d/'worker.log','w') as log:
            try:
                r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=c['process_limit_seconds'],env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                result=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else {}
                status=result.get('status','ERROR') if r.returncode==0 else 'ERROR';code=r.returncode
            except subprocess.TimeoutExpired:status='TIMEOUT';code=None
        write(d/'parent.json',{'status':status,'exit_code':code,'elapsed_seconds':time.monotonic()-t,'counts_towards_budget':True})
        print(method,seed,status,round(time.monotonic()-t,2),flush=True)
    evaluate(c,out,args.device)
    write(out/'WALL_COST.json',{'stage_wall_seconds':time.monotonic()-start,'includes':['data generation','preflight if GPU','all workers and failures','evaluation'],
                                 'excludes':['implementation','literature review','independent audit','packaging']})
    return 0

def verify(args):
    f=ROOT/'MANIFEST_SHA256.json'
    if not f.exists():raise FileNotFoundError(f)
    d=json.loads(f.read_text());bad=[k for k,v in d.items() if not (ROOT/k).exists() or sha(ROOT/k)!=v]
    print(json.dumps({'files':len(d),'mismatches':bad}));return int(bool(bad))

def collect(args):
    run=Path(args.run).resolve();out=Path(args.out).resolve()
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    files=[p for p in run.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    manifest={str(p.relative_to(run)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'run/'+str(p.relative_to(run)))
        for folder in ('pi2','pi4','scripts','configs','docs'):
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in str(p):z.write(p,'source/'+str(p.relative_to(ROOT)))
        z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,indent=2))
    print(out,sha(out));return 0

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    a=sp.add_parser('run');a.add_argument('--stage',choices=['cpu','gpu'],required=True);a.add_argument('--device',required=True);a.add_argument('--out',required=True)
    a=sp.add_parser('worker');a.add_argument('--config',required=True);a.add_argument('--data',required=True);a.add_argument('--method',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--out',required=True);a.add_argument('--device',required=True);a.add_argument('--started',type=float)
    sp.add_parser('verify')
    a=sp.add_parser('collect');a.add_argument('--run',required=True);a.add_argument('--out',required=True)
    args=p.parse_args();return {'run':run,'worker':worker,'verify':verify,'collect':collect}[args.command](args)
if __name__=='__main__':sys.exit(main())
