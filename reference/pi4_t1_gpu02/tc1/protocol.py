from __future__ import annotations
import json,random,hashlib,math
from pathlib import Path
from .budget import choose_checkpoint,fit_complete
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def source_manifest():
    paths=[]
    for folder in ('pi2','pi4','bc1','tc1','configs','scripts','docs','tests','provenance'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    paths.extend(ROOT/n for n in ('pyproject.toml','README_CN.md','AGENTS.md','CODEX_START_CN.md') if (ROOT/n).exists())
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def verify_frozen():
    old=read(ROOT/'provenance/B1_FROZEN_PYTHON_SHA256.json')
    bad=[k for k,v in old.items() if not (ROOT/k).is_file() or sha(ROOT/k)!=v]
    return {'status':'PASS' if not bad else 'FAIL','files':len(old),'mismatches':bad}

def validate(p):
    prior=read(ROOT/'configs/challenge.json')
    if p['methods']!=prior['methods']:raise ValueError('All six B1 methods required')
    if p['numerics']!=prior['numerics'] or p['decision']!=prior['decision']:
        raise ValueError('B1 score and fidelity gates must not change')
    lock=read(ROOT/'provenance/B1_HYPERPARAMETER_LOCK.json')
    if p['learning_rates']!=lock['learning_rates']:raise ValueError('B1 LR lock altered')
    if 'development' in p:raise ValueError('No new search allowed')
    if 'updates' in p['fit'] or 'validate_every' in p['fit']:raise ValueError('No step-based formal budget')
    t=p['time_policy'];f=p['fit']
    if t['primary_seconds']!=f['process_limit_seconds']:raise ValueError('Dual process clocks/limits')
    if not (0<f['cleanup_reserve_seconds']<f['process_limit_seconds']):raise ValueError('Bad reserve')
    if not all(0<b<t['primary_seconds'] for b in t['secondary_seconds']):raise ValueError('Bad secondary budget')
    if t['secondary_seconds']!=sorted(set(t['secondary_seconds'])):raise ValueError('Duplicate/unsorted budgets')
    if p['stage']=='gpu':
        keys=('history','horizon','reset_horizon','support_count','support_length','train_count','val_count','test_count',
              'general_batch','pair_batch','learning_rate','weight_decay','width','history_width',
              'difference_weight','consistency_weight','train_waits','test_waits','precision')
        if any(f[k]!=prior['fit'][k] for k in keys):raise ValueError('Scientific settings changed')
        if len(p['datasets'])!=3 or len(p['initialization_seeds'])!=3:raise ValueError('3 datasets x3 initializations required')
        if t['cpu_smoke_max_updates'] is not None:raise ValueError('CPU cap on GPU run')
    ids=[d['id'] for d in p['datasets']];seeds=[d['data_seed'] for d in p['datasets']]
    if len(set(ids))!=len(ids) or len(set(p['initialization_seeds']))!=len(p['initialization_seeds']):raise ValueError('Duplicate IDs/seeds')
    if any(abs(a-b)<1000 for i,a in enumerate(seeds) for b in seeds[i+1:]):raise ValueError('Overlapping random ranges')
    oldseeds=[prior['development']['data_seed']]+[d['data_seed'] for d in prior['datasets']]+[921101,922101,923101]
    if any(abs(a-b)<1000 for a in seeds for b in oldseeds):raise ValueError('Historical data range reused')


def jobs(p):
    """Balanced rotated ordering within dataset/seed blocks, not parallel fitting."""
    blocks=[(d['id'],s) for d in p['datasets'] for s in p['initialization_seeds']]
    random.Random(p['schedule_seed']).shuffle(blocks)
    order=list(p['methods']);random.Random(p['schedule_seed']+1).shuffle(order);out=[]
    for i,(ds,s) in enumerate(blocks):
        methods=order[i%len(order):]+order[:i%len(order)]
        for m in methods:
            out.append(dict(dataset=ds,trial_id=f'{m}_s{s}',method=m,seed=s,learning_rate=p['learning_rates'][m],
                            phase='time_comparison',complete_fit_budget_seconds=p['time_policy']['primary_seconds']))
    return out


def config_for(p,d):
    return dict(p['fit'],stage=p['stage'],data_seed=d['data_seed'],methods=p['methods'],seeds=p['initialization_seeds'],time_policy=p['time_policy'])


def check_trial(td:Path,p):
    try:
        pa=read(td/'parent.json');r=read(td/'result.json');c=read(td/'curve.json')
        if not fit_complete(pa,r,p['time_policy']['primary_seconds'],p['stage']):return 'incomplete/timeout/budget violation'
        if r.get('clock_origin_monotonic')!=pa.get('parent_origin_monotonic'):return 'parent/worker origin mismatch'
        if r['worker_elapsed']>pa['elapsed_seconds']:return 'worker time exceeds complete parent time'
        best=choose_checkpoint(c,p['time_policy']['primary_seconds'])
        if best is None:return 'no budget-qualified checkpoint'
        if sha(td/'best.pt')!=r['checkpoint_sha256'] or sha(td/best['checkpoint_path'])!=r['checkpoint_sha256']:
            return 'best hash/selection mismatch'
        if r['best_step']!=best['step'] or r['validation_score']!=best['selection_score']:return 'best score/step mismatch'
        for row in c:
            if row['checkpoint_ready_elapsed']>r['worker_elapsed']:return 'checkpoint completed after worker finalized'
            if sha(td/row['checkpoint_path'])!=row['checkpoint_sha256']:return 'recorded checkpoint hash mismatch'
    except Exception as e:return repr(e)
    return None


def prerequisites(run,p):
    run=Path(run);bad=[]
    for j in jobs(p):
        e=check_trial(run/'datasets'/j['dataset']/'trials'/j['trial_id'],p)
        if e:bad.append(dict(job=j,error=e))
    return bad
