#!/usr/bin/env python3
from __future__ import annotations
import os,sys,argparse,json,time,traceback
from pathlib import Path
# A single threaded CPU reference in each process; leave device assignment untouched.
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from c1.runner import run_stage,task,collect,verify_package
    p=argparse.ArgumentParser(description='PI-4 C1 independent confirmation; no model changes')
    sp=p.add_subparsers(dest='cmd',required=True)
    sp.add_parser('verify')
    a=sp.add_parser('run');a.add_argument('--stage',choices=['gpu','cpu_smoke'],required=True);a.add_argument('--device',required=True);a.add_argument('--out',type=Path,required=True)
    a=sp.add_parser('collect');a.add_argument('--run',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sp.add_parser('_task');a.add_argument('--action',choices=['prepare','preflight','fit','evaluate','audit','finalize'],required=True)
    a.add_argument('--run',type=Path,required=True);a.add_argument('--dataset');a.add_argument('--method');a.add_argument('--seed',type=int);a.add_argument('--started',type=float);a.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.cmd=='verify':
        r=verify_package();print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['status']=='PASS' else 3
    if args.cmd=='run':
        if verify_package()['status']!='PASS':raise RuntimeError('Delivery manifest or frozen core mismatch')
        return run_stage(args.stage,args.device,args.out)
    if args.cmd=='collect':print(json.dumps(collect(args.run,args.out),ensure_ascii=False,indent=2));return 0
    try:task(args.action,args.run,args.dataset,args.method,args.seed,args.device,args.started)
    except Exception:
        print(traceback.format_exc(),file=sys.stderr);return 1
    return 0
if __name__=='__main__':sys.exit(main())
