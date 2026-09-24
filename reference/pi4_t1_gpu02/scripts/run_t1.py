#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    s.add_parser('verify')
    r=s.add_parser('run');r.add_argument('--stage',choices=['cpu','gpu'],required=True);r.add_argument('--device');r.add_argument('--out',required=True)
    c=s.add_parser('collect');c.add_argument('--run',required=True);c.add_argument('--out',required=True)
    t=s.add_parser('_task');t.add_argument('--action',required=True);t.add_argument('--run',required=True);t.add_argument('--device',required=True);t.add_argument('--dataset');t.add_argument('--trial');t.add_argument('--started',type=float)
    a=p.parse_args();from tc1 import runner
    if a.cmd=='verify':
        q=runner.verify_package();print(json.dumps(q,indent=2));return 0 if q['status']=='PASS' else 3
    if a.cmd=='collect':
        q=runner.collect(a.run,a.out);print(json.dumps(q,indent=2));return 3 if q['whole_stage_budget_exceeded_at_collection'] else 0
    if a.cmd=='run':return runner.run_stage(a.stage,a.device or ('cuda:0' if a.stage=='gpu' else 'cpu'),a.out)
    runner.task(a.action,a.run,a.device,a.dataset,a.trial,a.started);return 0
if __name__=='__main__':raise SystemExit(main())
