#!/usr/bin/env python3
"""Rebuild publication tables from frozen small JSON records. Standard library only.
No training, torch import, checkpoint loading, downloading, or scientific re-selection.
Default checks committed outputs; --write deliberately refreshes derived files only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('pi4', 'c1', 'b1', 't1')
CASE_KEYS = {('ring4',0),('ring8',0),('long8',0),('reset_iso',0),('reset_iso',256),
             ('reset_ring4',0),('reset_ring4',256),('reset_null',0),('reset_null',256)}
METHODS_5 = {'fixed_mse','fixed_behavior','free_mse','free_behavior','ode_behavior'}
METHODS_6 = {'fixed_behavior','free_behavior','gru4_behavior','gru32_behavior','graph32_behavior','node32_behavior'}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Geometric mean requires nonempty finite positive values; do not silently drop zeros.')
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))


def ratio(a: Any,b: Any) -> float:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= 0:
        raise ValueError('Undefined scientific ratio; keep nulls or inspect the protocol rather than filling with zero.')
    return float(a/b)


def verify_sources(root: Path) -> int:
    manifest = read_json(root/'results/evidence_manifest.json')
    seen=set()
    for e in manifest['entries']:
        rel=Path(e['path'])
        if rel.is_absolute() or '..' in rel.parts or e['path'] in seen:
            raise ValueError(f'Unsafe/duplicate manifest entry {rel}')
        seen.add(e['path'])
        p=root/rel
        if p.is_symlink() or not p.is_file(): raise ValueError(f'Missing/non-regular evidence {rel}')
        raw=p.read_bytes()
        if len(raw)!=e['size'] or hashlib.sha256(raw).hexdigest()!=e['sha256']:
            raise ValueError(f'Evidence digest mismatch: {rel}')
    return len(seen)


def load_records(root: Path) -> list[dict[str,Any]]:
    all_rows=[]
    for stage in STAGES:
        base=root/'results/source_records'/stage/'run'
        files=[base/'summary.json'] if stage=='pi4' else sorted(base.glob('datasets/*/summary.json'))
        if len(files)!=(1 if stage=='pi4' else 3): raise ValueError(f'{stage}: incomplete datasets')
        expected=METHODS_5 if stage in ('pi4','c1') else METHODS_6
        for file in files:
            d=read_json(file);dataset='screen' if stage=='pi4' else file.parent.name
            records=d['records'];methods={r['method'] for r in records}
            if methods!=expected: raise ValueError(f'{stage}/{dataset}: missing or extra methods {methods ^ expected}')
            seeds={r['seed'] for r in records}
            if len(seeds)!=3: raise ValueError('Expected three paired initializations, not independent datasets.')
            keys=[(r['method'],r['seed'],r['case'],r['wait']) for r in records]
            if len(keys)!=len(set(keys)):raise ValueError(f'Duplicate results {file}')
            wanted={(m,s,c,w) for m in methods for s in seeds for c,w in CASE_KEYS}
            if set(keys)!=wanted:raise ValueError(f'{stage}/{dataset}: missing or unexpected scenario rows')
            for i,r in enumerate(records):
                if r['case']=='reset_null' and r.get('difference_error') is not None:
                    raise ValueError('Zero-difference boundary must remain undefined, not zero performance.')
                all_rows.append({'study':stage,'dataset':dataset,**r,
                                 'source_file':file.relative_to(root).as_posix(),'source_pointer':f'/records/{i}'})
    return all_rows


def compare(rows: list[dict[str,Any]],baseline: str) -> dict[str,Any]:
    seeds=sorted({r['seed'] for r in rows})
    keyed={(r['method'],r['seed'],r['case'],r['wait']):r for r in rows}
    seed_ratios=[]
    for seed in seeds:
        seed_ratios.append(geomean([ratio(keyed[('fixed_behavior',seed,c,256)]['difference_error'],
                                         keyed[(baseline,seed,c,256)]['difference_error']) for c in ('reset_iso','reset_ring4')]))
    mechanism=geomean(seed_ratios)
    general={c:statistics.median([ratio(keyed[('fixed_behavior',s,c,0)]['nmse'],keyed[(baseline,s,c,0)]['nmse']) for s in seeds]) for c in ('ring4','ring8','long8')}
    return {'mechanism_ratio':mechanism,**{c+'_ratio':v for c,v in general.items()},
            'seed_mechanism_ratios':seed_ratios,
            'candidate_gate':mechanism<=0.9 and all(v<=1.05 for v in general.values()) and sum(v<1 for v in seed_ratios)>=2}


def check_recorded(root: Path,stage: str,dataset: str,baseline: str,x: dict[str,Any]) -> None:
    if stage=='pi4':return # Preserve its two-stage numerical/validity statuses; do not restamp the legacy receipt.
    d=read_json(root/'results/source_records'/stage/'run/DECISION.json')
    rec=next(r for r in d['datasets'] if r['dataset']==dataset)
    if stage=='c1':
        if baseline!='free_behavior':return
        rec=rec['saved'];m=rec['mechanism_geomean_ratio'];g=rec['general_paired_median_ratios'];p=rec['passed']
    else:
        rec=next(r for r in rec['comparisons'] if r['baseline']==baseline)
        m=rec['candidate_mechanism_ratio'];g=rec['candidate_general_median_ratios'];p=rec['candidate_passes_original_gate']
    if not math.isclose(m,x['mechanism_ratio'],rel_tol=1e-11,abs_tol=1e-13):raise ValueError(f'{stage}/{dataset}: mechanism mismatch')
    for c,v in g.items():
        if not math.isclose(v,x[c+'_ratio'],rel_tol=1e-11,abs_tol=1e-13):raise ValueError(f'{stage}/{dataset}/{c}: general mismatch')
    if p != x['candidate_gate']:raise ValueError(f'{stage}/{dataset}: gate mismatch')


def csv_text(rows: list[dict[str,Any]],columns: list[str]) -> str:
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');w.writeheader()
    for row in rows:w.writerow({k:('' if row.get(k) is None else row.get(k)) for k in columns})
    return f.getvalue()


def generate(root: Path=ROOT) -> dict[str,str]:
    n=verify_sources(root);rows=load_records(root);comparisons=[]
    for stage in STAGES:
        for dataset in sorted({r['dataset'] for r in rows if r['study']==stage}):
            subset=[r for r in rows if r['study']==stage and r['dataset']==dataset]
            bs=['free_behavior'] if stage in ('pi4','c1') else sorted(METHODS_6-{'fixed_behavior'})
            for b in bs:
                v=compare(subset,b);check_recorded(root,stage,dataset,b,v)
                comparisons.append({'study':stage,'dataset':dataset,'baseline':b,**v})
    absolute=[]
    for stage in STAGES:
        sub=[r for r in rows if r['study']==stage]
        for dataset in sorted({r['dataset'] for r in sub}):
            for method in sorted({r['method'] for r in sub}):
                for case,wait in sorted(CASE_KEYS):
                    ss=[r for r in sub if r['dataset']==dataset and r['method']==method and r['case']==case and r['wait']==wait]
                    out={'study':stage,'dataset':dataset,'method':method,'case':case,'wait':wait,'n_initializations':len(ss)}
                    for col in ('mse','nmse','difference_error','repeat_mse','wrong_ratio','repeat_vs_difference'):
                        vs=[r.get(col) for r in ss]
                        out[col+'_median']=None if all(v is None for v in vs) else statistics.median(vs)
                    absolute.append(out)
    status=[]
    for stage in STAGES:
        base=root/'results/source_records'/stage/'run'
        if stage=='pi4':
            d=read_json(base/'execution_status_public.json')
            status.append({'study':stage,'scientific_numeric_status':d['raw_numeric_decision'],'original_validity_status':d['decision_validity'],'note':'Legacy replay stopped at elementwise tolerance; later supplied independent review supports task-level result, without changing this original validity field.'})
        else:
            d=read_json(base/'DECISION.json')
            status.append({'study':stage,'scientific_numeric_status':d['status'],'original_validity_status':'see per-dataset audit fields in original DECISION.json','note':d.get('scope','')})
    wall=read_json(root/'results/source_records/t1/run/WALL_COST.json');cost=[]
    for fit in wall['fits']:
        rel=f"results/source_records/t1/run/datasets/{fit['dataset']}/trials/{fit['trial_id']}/result.json"
        result=read_json(root/rel)
        parent_path=root/rel.replace('/result.json','/parent.json')
        parent=read_json(parent_path)
        if result['status']!='COMPLETE' or parent.get('status')!='COMPLETE':raise ValueError('Incomplete T1 fit; do not silently exclude it.')
        cost.append({'study':'t1','dataset':fit['dataset'],'method':fit['method'],'seed':fit['seed'],
                     'updates':result['updates'],'parameters':result['parameters'],'persistent_dimension':result['persistent_dimension'],
                     'elapsed_seconds':fit['elapsed_seconds'],'cap_seconds':600,'validation_calls':result['validation_calls'],
                     'source_file':rel,'clock_source':'results/source_records/t1/run/WALL_COST.json'})
    if len(cost)!=54:raise ValueError('Expected all 54 fits for cost table')
    cost_summary=[]
    for method in sorted(METHODS_6):
        ss=[r for r in cost if r['method']==method]
        out={'method':method,'n_fits':len(ss)}
        for col in ('parameters','persistent_dimension','updates','elapsed_seconds','validation_calls'):
            out[col+'_median']=statistics.median(r[col] for r in ss)
        cost_summary.append(out)
    wins=[]
    for b in sorted(METHODS_6-{'fixed_behavior'}):
        for category,cases in [('mechanism',[('reset_iso',256),('reset_ring4',256)]),('short',[('ring4',0),('ring8',0)]),('long',[('long8',0)])]:
            rr=[]
            for dataset in ('t01','t02','t03'):
                keyed={(r['method'],r['seed'],r['case'],r['wait']):r for r in rows if r['study']=='t1' and r['dataset']==dataset}
                for seed in sorted({k[1] for k in keyed}):
                    for case,wait in cases:
                        field='difference_error' if wait else 'nmse'
                        rr.append(ratio(keyed[('fixed_behavior',seed,case,wait)][field],keyed[(b,seed,case,wait)][field]))
            wins.append({'study':'t1','baseline':b,'category':category,'candidate_lower_count':sum(r<1 for r in rr),'n_paired_comparisons':len(rr),'note':'Descriptive paired counts, NOT independent dataset count or a significance test.'})
    md=['# Rebuilt result tables','',
        'Generated by `python scripts/rebuild_results.py --write` from the committed JSON records. No weights were loaded and no new experiments were run.',
        'All ratios are candidate / named comparator: below 1 favors the candidate. Do not join studies into a causal training-progress curve.','',
        '## Frozen stage decisions','', '| Study | Original numeric decision | Original validity |','|---|---|---|']
    for r in status:md.append(f"| {r['study']} | `{r['scientific_numeric_status']}` | {r['original_validity_status']} |")
    for stage in ('c1','b1','t1'):
        md += ['',f'## {stage.upper()}: all prescribed comparisons','', '| Dataset | Comparator | Mechanism GM | ring4 median ratio | ring8 median ratio | long8 median ratio | Candidate full gate |','|---|---|---:|---:|---:|---:|---|']
        for r in comparisons:
            if r['study']==stage:
                md.append(f"| {r['dataset']} | {r['baseline']} | {r['mechanism_ratio']:.6f} | {r['ring4_ratio']:.6f} | {r['ring8_ratio']:.6f} | {r['long8_ratio']:.6f} | {r['candidate_gate']} |")
    md+=['','## T1 recorded complete-fit costs','', 'All methods had the same 600-second upper bound, including process setup and exit, with a 30-second reserve. This is not an equal-parameter ablation or a deployment latency benchmark.','', '| Method | Parameters | Persistent scalars/component | Updates median | Complete seconds median |','|---|---:|---:|---:|---:|']
    for r in cost_summary:md.append(f"| {r['method']} | {r['parameters_median']:g} | {r['persistent_dimension_median']:g} | {r['updates_median']:g} | {r['elapsed_seconds_median']:.3f} |")
    md+=['','## T1 directional counts','', '| Comparator | Task | Candidate lower-error pairs |','|---|---|---:|']
    for r in wins:md.append(f"| {r['baseline']} | {r['category']} | {r['candidate_lower_count']}/{r['n_paired_comparisons']} |")
    md+=['','Pairs share three independently generated datasets and three initialization seeds per dataset. They are not independent external tasks.',
         'Zero-target-difference scenarios retain null normalized scores. See `absolute_scores.csv` for all methods and all nine scenario/wait combinations.',
         'The original PI-4 replay failure and later task-level replay support are separate facts; this table generator does not re-validate checkpoint numerics.','']
    report={'source_files_checked':n,'source_rows':len(rows),'comparison_rows':len(comparisons),'t1_fit_rows':len(cost),'recorded_decision_comparisons_consistent':True,
            'scope':'Small JSON integrity, all-record aggregation and matching against recorded decisions only. No model replay, no scientific replication, no numerical tolerance reclassification.'}
    fields=['study','dataset','method','seed','case','kind','wait','mse','nmse','difference_error','difference_target_energy','difference_output_energy','repeat_mse','wrong_mse','wrong_ratio','repeat_vs_difference','source_file','source_pointer']
    pairfields=['study','dataset','baseline','mechanism_ratio','ring4_ratio','ring8_ratio','long8_ratio','candidate_gate']
    return {
      'results/derived/all_scores.csv':csv_text(rows,fields),
      'results/derived/paired_ratios.csv':csv_text(comparisons,pairfields),
      'results/derived/absolute_scores.csv':csv_text(absolute,list(absolute[0])),
      'results/derived/t1_fit_costs.csv':csv_text(cost,list(cost[0])),
      'results/derived/t1_cost_summary.csv':csv_text(cost_summary,list(cost_summary[0])),
      'results/derived/t1_directional_counts.csv':csv_text(wins,list(wins[0])),
      'results/derived/study_statuses.json':json.dumps(status,ensure_ascii=False,indent=2)+'\n',
      'results/derived/TABLES.md':'\n'.join(md),
      'results/derived/REBUILD_CHECK.json':json.dumps(report,ensure_ascii=False,indent=2)+'\n'}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true',help='Rewrite derived outputs only. Default is a read-only check.');args=p.parse_args()
    try:
        generated=generate();mismatch=[]
        for rel,s in generated.items():
            f=ROOT/rel
            if args.write:f.parent.mkdir(parents=True,exist_ok=True);f.write_text(s,encoding='utf-8')
            elif not f.is_file() or f.read_text(encoding='utf-8')!=s:mismatch.append(rel)
        if mismatch:raise ValueError('Derived outputs missing or stale: '+', '.join(mismatch))
        print(json.dumps({'status':'PASS','action':'write' if args.write else 'check','derived_files':len(generated)},indent=2));return 0
    except (OSError,ValueError,KeyError,StopIteration) as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)}),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
