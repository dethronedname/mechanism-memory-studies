"""Per-dataset, per-comparator outcomes. Never use test results to choose LR."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from pi4.io import write
from .protocol import read
RESET=('reset_iso','reset_ring4');GENERAL=('ring4','ring8','long8')

def gm(x):
    a=np.asarray(x,dtype=float)
    if not a.size or not np.isfinite(a).all() or (a<=0).any():raise ValueError('Invalid positive ratio; not epsilon-filled')
    return float(np.exp(np.log(a).mean()))

def comparison(records,seeds,wait,base,t):
    keys=[(r['method'],r['seed'],r['case'],r['wait']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate score rows')
    tab=dict(zip(keys,records))
    def vals(a,b,case,metric,w):return [tab[(a,s,case,w)][metric]/tab[(b,s,case,w)][metric] for s in seeds]
    a='fixed_behavior';b=base
    mech=[vals(a,b,c,'difference_error',wait) for c in RESET]
    generals={c:float(np.median(vals(a,b,c,'mse',0))) for c in GENERAL}
    seedrat=[gm(v) for v in np.asarray(mech).T];m=gm(mech)
    candidate=(m<=t['mechanism_geomean_max'] and all(v<=t['each_general_median_max'] for v in generals.values()) and sum(v<1 for v in seedrat)>=min(t['seed_wins_min'],len(seeds)))
    # Predeclared descriptive replication margin, not statistical equivalence test.
    reverse_general={c:float(np.median(vals(b,a,c,'mse',0))) for c in GENERAL}
    matched=(1/m<=1.05 and all(v<=1.05 for v in reverse_general.values()))
    boundary=abs(m-t['mechanism_geomean_max'])<=t['numeric_boundary_guard'] or any(abs(v-t['each_general_median_max'])<=t['numeric_boundary_guard'] for v in generals.values())
    boundary=boundary or abs(1/m-1.05)<=t['numeric_boundary_guard'] or any(abs(v-1.05)<=t['numeric_boundary_guard'] for v in reverse_general.values())
    return {'baseline':b,'candidate_mechanism_ratio':m,'candidate_general_median_ratios':generals,'seed_ratios':seedrat,
            'candidate_passes_original_gate':bool(candidate),'baseline_matches_within_5pct':bool(matched),'boundary_sensitive':bool(boundary),
            'warning':'Three paired initializations within one dataset; neither model-family SOTA nor equivalence proof.'}

def finalize(run,p):
    run=Path(run);groups=[];problems=[]
    for ds in p['datasets']:
        root=run/'datasets'/ds['id']
        try:
            s=read(root/'summary.json');a=read(root/'audit.json');comparisons=[]
            if a['status']!='PASS':problems.append({'dataset':ds['id'],'reason':'Replay/integrity failed'})
            for base in p['methods'][1:]:
                old=comparison(s['records'],p['initialization_seeds'],max(p['fit']['test_waits']),base,p['decision'])
                new=comparison(a['replayed_records'],p['initialization_seeds'],max(p['fit']['test_waits']),base,p['decision'])
                flags=('candidate_passes_original_gate','baseline_matches_within_5pct')
                if any(old[k]!=new[k] for k in flags) or (p['stage']=='gpu' and (old['boundary_sensitive'] or new['boundary_sensitive'])):
                    problems.append({'dataset':ds['id'],'baseline':base,'reason':'Numerical decision sensitivity'})
                comparisons.append(old)
            groups.append({'dataset':ds['id'],'comparisons':comparisons,'audit_status':a['status'],'legacy_status':a['legacy_status']})
        except Exception as e:problems.append({'dataset':ds['id'],'error':repr(e)})
    if p['stage']=='cpu':status='CPU_ENGINEERING_ONLY_NOT_SCIENTIFIC' if not problems else 'CPU_ENGINEERING_FAIL'
    elif problems or len(groups)!=len(p['datasets']):status='INCONCLUSIVE'
    else:
        all_win=all(r['candidate_passes_original_gate'] for g in groups for r in g['comparisons'])
        reproduced=[base for base in p['methods'][2:] if all(next(r for r in g['comparisons'] if r['baseline']==base)['baseline_matches_within_5pct'] for g in groups)]
        status='LIMITED_COMPETITIVE_SIGNAL' if all_win else ('BASELINE_REPRODUCES_SIGNAL' if reproduced else 'MIXED_TRADEOFF_NO_DOMINANCE')
    out={'protocol':p['protocol'],'status':status,'datasets':groups,'problems':problems,'automatic_next_experiment':False,
        'scope':'Same simulator, task-adapted baseline implementations, not claims against official GNS/GRU/NODE methods as a whole',
        'historical_C1':'PARTIAL_REPLICATION unchanged','cpu_is_performance_evidence':False}
    write(run/'DECISION.json',out)
    lines=['# PI-4 B1 基线挑战结果', '',f'状态：`{status}`。旧C1部分重复不改写。','',
      '这是任务适配对照比较，不是官方模型完整复现或通用架构排名。各模型同更新数，不是严格同墙钟。', '',
      '## 每套数据、每个对照', '|数据|对照|候选/对照机制比|ring4|ring8|long8|候选完整门槛|', '|---|---|---:|---:|---:|---:|---|']
    for g in groups:
        for r in g['comparisons']:
            v=r['candidate_general_median_ratios'];lines.append(f"|{g['dataset']}|{r['baseline']}|{r['candidate_mechanism_ratio']:.5f}|{v['ring4']:.5f}|{v['ring8']:.5f}|{v['long8']:.5f}|{r['candidate_passes_original_gate']}|")
    lines+=['','## 成本与信息利用','完整summary保存全部模型的绝对误差、同机制换校准误差和错机制校准误差。不能把更高误差基线上的比例改善当作准确机制辨识。',
      '每trial结果含参数量、持久状态、峰值CUDA allocator内存、训练净时与父进程时间。学习率开发成本另列；不宣称加速。', '', '## 问题记录',str(problems), '', '本轮不自动补跑、改候选或扩模。']
    (run/'REPORT_CN.md').write_text('\n'.join(lines),encoding='utf8');return out
