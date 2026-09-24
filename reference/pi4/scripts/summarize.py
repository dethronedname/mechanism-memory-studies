from __future__ import annotations
import sys,json,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi4.io import write

def run(runpath,out):
    r=Path(runpath);s=json.loads((r/'summary.json').read_text());cfg=json.loads((r/'config.json').read_text());rows=s['records'];groups={}
    for x in rows:groups.setdefault((x['method'],x['case'],x['wait']),[]).append(x)
    med=[]
    for (method,case,wait),g in groups.items():
        rec={'method':method,'case':case,'wait':wait,'n_seeds':len(g)}
        for key in ['mse','nmse','difference_error','repeat_mse','wrong_ratio','repeat_vs_difference','difference_amplitude_ratio']:
            vals=[a[key] for a in g if a.get(key) is not None]
            rec[key]=float(np.median(vals)) if vals else None
        med.append(rec)
    def pairing(a,b,case,wait,metric):
        ra={x['seed']:x for x in groups[(a,case,wait)]};rb={x['seed']:x for x in groups[(b,case,wait)]}
        ratios=[ra[sd][metric]/rb[sd][metric] for sd in cfg['seeds']]
        return {'a':a,'b':b,'case':case,'wait':wait,'metric':metric,'ratios':ratios,'median':float(np.median(ratios)),'geomean':float(np.exp(np.mean(np.log(ratios)))),'wins':sum(v<1 for v in ratios)}
    comparisons=[]
    for a,b in [('fixed_behavior','fixed_mse'),('free_behavior','free_mse'),('fixed_behavior','free_behavior'),('fixed_behavior','ode_behavior')]:
        for case in ['ring4','ring8','long8']:comparisons.append(pairing(a,b,case,0,'mse'))
        for case in ['reset_iso','reset_ring4']:
            for wait in cfg['test_waits']:comparisons.append(pairing(a,b,case,wait,'difference_error'))
    g=[x for x in comparisons if x['a']=='fixed_behavior' and x['b']=='free_behavior']
    resets=[x for x in g if x['metric']=='difference_error' and x['wait']==256];general=[x for x in g if x['metric']=='mse']
    allr=[v for x in resets for v in x['ratios']];gm=float(np.exp(np.mean(np.log(allr))))
    seed_gm=np.exp(np.mean(np.log(np.array([x['ratios'] for x in resets])),axis=0))
    passed=gm<=.9 and all(x['median']<=1.05 for x in general) and int((seed_gm<1).sum())>=2
    result={'medians':med,'paired_comparisons':comparisons,'decision':{'status':'LIMITED_ARCHITECTURE_SIGNAL' if passed else 'NO_CONFIRMED_ARCHITECTURE_ADVANTAGE',
            'mechanism_wait_geomean_ratio':gm,'general_ratios':[x['median'] for x in general],
            'seed_mechanism_geomean_ratios':seed_gm.tolist(),'frozen_thresholds':{'delta_geomean_max':.9,'each_general_median_max':1.05,'seed_wins_min':2},
            'scope':'finite CPU screen; no statistical confirmation and no automated further run'},'raw_status':s['status']}
    write(out,result);return result
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
