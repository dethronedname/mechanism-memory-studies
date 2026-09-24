"""Frozen study schedule. New baselines only: select two LRs on separate dev data."""
from __future__ import annotations
import json,random
from pathlib import Path
from pi4.io import sha,write
from .models import METHODS,NEW_METHODS
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def source_manifest():
    paths=[]
    for folder in ('pi2','pi4','bc1','configs','scripts','docs','tests'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    paths.extend(ROOT/n for n in ('pyproject.toml','AGENTS.md','README_CN.md','CODEX_START_CN.md') if (ROOT/n).exists())
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def verify_core():
    frozen=read(ROOT/'provenance/ORIGINAL_CORE_SHA256.json')
    bad=[k for k,h in frozen.items() if not (ROOT/k).is_file() or sha(ROOT/k)!=h]
    return {'status':'PASS' if not bad else 'FAIL','files':len(frozen),'mismatches':bad}
def validate(p):
    if p['methods']!=list(METHODS):raise ValueError('Method coverage/order changed')
    if p['development']['methods']!=list(NEW_METHODS):raise ValueError('Development family coverage changed')
    if p['development']['learning_rates']!=[.001,.003]:raise ValueError('Frozen finite LR search altered')
    if p['stage']=='gpu':
        ref=read(ROOT/'configs/legacy_gpu.json')
        keys=('history','horizon','reset_horizon','support_count','support_length','train_count','val_count','test_count',
              'updates','validate_every','general_batch','pair_batch','learning_rate','weight_decay','width','history_width',
              'difference_weight','consistency_weight','train_waits','test_waits','precision')
        if any(p['fit'][k]!=ref[k] for k in keys):raise ValueError('Candidate or scientific training settings changed')
        if len(p['datasets'])!=3 or len(p['initialization_seeds'])!=3:raise ValueError('Exactly 3 fresh datasets x 3 seeds')
    seeds=[p['development']['data_seed']]+[x['data_seed'] for x in p['datasets']]
    if any(abs(a-b)<1000 for i,a in enumerate(seeds) for b in seeds[i+1:]):raise ValueError('Overlapping data RNG ranges')
def development_jobs(p):
    jobs=[]
    for m in p['development']['methods']:
        for index,lr in enumerate(p['development']['learning_rates']):
            jobs.append(dict(dataset='dev',trial_id=f'{m}_r{index}',method=m,seed=p['development']['initialization_seed'],learning_rate=lr,phase='development'))
    random.Random(p['schedule_seed']).shuffle(jobs);return jobs

def confirmation_jobs(p):
    jobs=[]
    for s in p['initialization_seeds']:
        block=[dict(dataset=d['id'],trial_id=f'{m}_s{s}',method=m,seed=s,phase='confirmation') for d in p['datasets'] for m in p['methods']]
        random.Random(p['schedule_seed']+s).shuffle(block);jobs.extend(block)
    return jobs

def config_for(p,d,dev=False):
    c=dict(p['fit'],data_seed=d['data_seed'],methods=p['methods'],seeds=p['initialization_seeds'])
    if dev:c['seeds']=[p['development']['initialization_seed']]
    return c

def check_trial(td,updates):
    try:
        pa=read(td/'parent.json');r=read(td/'result.json')
        if pa['status']!='COMPLETE' or r['status']!='COMPLETE' or r['updates']!=updates:return 'incomplete'
        if r['checkpoint_sha256']!=sha(td/'best.pt'):return 'checkpoint hash'
    except Exception as e:return repr(e)
    return None

def select_development(run,p):
    """Reads only fit result.json computed on DEV validation. No test row accepted."""
    run=Path(run)
    if (run/'HYPERPARAMETER_LOCK.json').exists() or (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('No reselection')
    allrows=[]
    for j in development_jobs(p):
        td=run/'development/trials'/j['trial_id'];bad=check_trial(td,p['fit']['updates'])
        if bad:raise RuntimeError('Incomplete development: '+j['trial_id']+' '+bad)
        r=read(td/'result.json')
        allrows.append(dict(j,validation_score=r['validation_score'],result_sha256=sha(td/'result.json')))
    rates={m:p['fit']['learning_rate'] for m in p['original_methods']};selected={}
    for m in p['development']['methods']:
        rows=[r for r in allrows if r['method']==m]
        if len(rows)!=2:raise ValueError('Two LR results required')
        best=min(rows,key=lambda r:(r['validation_score'],r['learning_rate']))
        rates[m]=best['learning_rate'];selected[m]=best
    out={'status':'LOCKED','scope':'DEV validation only; new families may tune LR, originals unchanged',
         'learning_rates':rates,'selected':selected,'all_development_results':allrows}
    write(run/'HYPERPARAMETER_LOCK.json',out);return out

def prerequisites(run,p):
    bad=[]
    if not (run/'HYPERPARAMETER_LOCK.json').exists():bad.append('No LR lock')
    for j in confirmation_jobs(p):
        err=check_trial(run/'datasets'/j['dataset']/'trials'/j['trial_id'],p['fit']['updates'])
        if err:bad.append(dict(job=j,error=err))
    return bad
