"""Dataset-level replication; initialization/trajectory rows are not independent datasets."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from .protocol import read
from pi4.io import write
GENERAL=('ring4','ring8','long8')
RESET=('reset_iso','reset_ring4')
def gm(values):
    a=np.asarray(values,dtype=float)
    if a.size==0 or not np.isfinite(a).all() or (a<=0).any():
        raise ValueError('Undefined positive error ratio; do not silently epsilon-fill')
    return float(np.exp(np.log(a).mean()))

def dataset_gate(records:list[dict],seeds:list[int],wait:int,thresholds:dict)->dict:
    keys=[(r['method'],r['seed'],r['case'],r['wait']) for r in records]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate evaluation rows')
    tab=dict(zip(keys,records));a='fixed_behavior';b='free_behavior'
    def ratios(case,metric,w):
        return [tab[(a,s,case,w)][metric]/tab[(b,s,case,w)][metric] for s in seeds]
    rr=[ratios(c,'difference_error',wait) for c in RESET]
    mechanism=gm(np.ravel(rr));seed_ratios=[gm(v) for v in np.asarray(rr).T]
    gr={c:float(np.median(ratios(c,'mse',0))) for c in GENERAL}
    checks={'mechanism':mechanism<=thresholds['mechanism_geomean_max'],
      'general':all(x<=thresholds['each_general_median_max'] for x in gr.values()),
      'seed_direction':sum(x<1 for x in seed_ratios)>=thresholds['seed_wins_min']}
    # A numerical classification very close to a boundary is held inconclusive, not nudged.
    guard=thresholds['numeric_boundary_guard']
    boundary=(abs(mechanism-thresholds['mechanism_geomean_max'])<=guard or
       any(abs(x-thresholds['each_general_median_max'])<=guard for x in gr.values()) or
       any(abs(x-1)<=guard for x in seed_ratios))
    diagnostic={}
    for case in RESET:
        rows=[tab[(a,s,case,wait)] for s in seeds]
        diagnostic[case]={
           'same_mechanism_repeat_over_true_difference':[r['repeat_vs_difference'] for r in rows],
           'wrong_mechanism_mse_ratio':[r['wrong_ratio'] for r in rows],
           'candidate_difference_errors':[r['difference_error'] for r in rows]}
    # Descriptive log-scale 2x2 contrast, not a significance claim.
    contrast=[]
    for s in seeds:
        for c in RESET:
            e=lambda m:tab[(m,s,c,wait)]['difference_error']
            contrast.append(math.log(e('fixed_behavior')/e('fixed_mse'))-
                            math.log(e('free_behavior')/e('free_mse')))
    return {'passed':all(checks.values()),'checks':checks,'boundary_sensitive':boundary,
       'mechanism_geomean_ratio':mechanism,'general_paired_median_ratios':gr,
       'seed_mechanism_geomeans':seed_ratios,'mechanism_diagnostics':diagnostic,
       'log_2x2_contrast_mean':float(np.mean(contrast)),
       'warning':'Calibration consistency and mechanism reading are reported; no new post-hoc gate.'}

def aggregate(run:Path,p:dict)->dict:
    results=[];problems=[]
    for d in p['datasets']:
        path=run/'datasets'/d['id']
        try:
            summ=read(path/'summary.json');audit=read(path/'audit.json')
            saved=dataset_gate(summ['records'],p['initialization_seeds'],max(p['fit']['test_waits']),p['decision'])
            replay=dataset_gate(audit['replayed_records'],p['initialization_seeds'],max(p['fit']['test_waits']),p['decision'])
            row={'dataset':d['id'],'data_seed':d['data_seed'],'saved':saved,'cpu_replay':replay,
                 'audit_status':audit['status'],'legacy_elementwise_status':audit['legacy_status']}
            results.append(row)
            if audit['status']!='PASS':problems.append({'dataset':d['id'],'reason':'numerical/integrity audit failed'})
            if p['stage']=='gpu' and (saved['passed']!=replay['passed'] or saved['boundary_sensitive'] or replay['boundary_sensitive']):
                problems.append({'dataset':d['id'],'reason':'threshold classification numerically sensitive'})
        except Exception as e:problems.append({'dataset':d['id'],'reason':repr(e)})
    if p['stage']!='gpu':
        status='CPU_ENGINEERING_FAIL' if problems or len(results)!=len(p['datasets']) else 'CPU_ENGINEERING_ONLY_NOT_SCIENTIFIC'
    elif problems or len(results)!=len(p['datasets']):status='INCONCLUSIVE'
    else:
        passes=sum(r['saved']['passed'] for r in results)
        status=('REPLICATED_WITHIN_SIMULATOR' if passes==len(results)
                else 'PARTIAL_REPLICATION' if passes else 'NOT_REPLICATED')
    vals=[r['saved']['mechanism_geomean_ratio'] for r in results]
    out={'protocol':p['protocol'],'status':status,'datasets':results,'problems':problems,
      'n_independent_data_draws':len(results),'initializations_per_data_draw':len(p['initialization_seeds']),
      'dataset_pass_count':sum(r['saved']['passed'] for r in results),
      'pooled_equal_dataset_mechanism_geomean':gm(vals) if vals else None,
      'scope':'Replication within the same known simulator and parameter law. No cross-simulator, measured-data, significance, or architecture-originality claim.',
      'historical_status_unchanged':p['historical_status'],'automatic_next_experiment':False}
    write(run/'DECISION.json',out)
    lines=['# PI-4 C1 独立数据确认报告','',f"**状态：{status}**",'',
      '固定原模型、损失、1600步配方（CPU smoke除外）。每套数据单独检查原效应门槛；不将初始化或轨迹当成独立数据集。','',
      '| 数据 | 两项机制误差比几何平均 | 原门槛通过 | 数值审计 | 旧逐元素检查 |',
      '|---|---:|---|---|---|']
    for r in results:
        lines.append(f"| {r['dataset']} | {r['saved']['mechanism_geomean_ratio']:.6g} | {r['saved']['passed']} | {r['audit_status']} | {r['legacy_elementwise_status']} |")
    lines+=['','同机制换校准、错机制交换、一般预测、空差分边界和2×2对照详见各数据集 summary.json/audit.json 及 DECISION.json。',
      '即使主门槛全部通过，校准一致性较差时也只能称机制信息利用改善，不能称准确不变辨识。',
      'PARTIAL_REPLICATION / NOT_REPLICATED 不自动补种子、改损失或延长训练。INCONCLUSIVE 不是模型无收益。',
      'GPU速度不是本轮主终点；相同更新数并非相同墙钟。',
      '三套独立生成数据仍来自同一模拟器，不构成外部任务验证或统计显著性证明。',
      '', '问题记录：'+str(problems)]
    (run/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8');return out
