"""Protocol immutability, experiment schedule and strict prerequisite checks."""
from __future__ import annotations
import json, random
from pathlib import Path
from pi4.io import sha, write
ROOT=Path(__file__).resolve().parents[1]
# Training/model/data-generation settings below are inherited without changing values.
FROZEN_KEYS=('methods','history','horizon','reset_horizon','support_count','support_length',
 'train_count','val_count','test_count','updates','validate_every','general_batch','pair_batch',
 'learning_rate','weight_decay','width','history_width','difference_weight','consistency_weight',
 'train_waits','test_waits','process_limit_seconds','cleanup_reserve_seconds','precision','selection')
def read(path: str|Path):
    return json.loads(Path(path).read_text(encoding='utf8'))
def validate(p: dict)->None:
    if p['stage']=='gpu':
        old=read(ROOT/'configs/gpu.json')
        changes=[k for k in FROZEN_KEYS if p['fit'][k]!=old[k]]
        if changes: raise ValueError('Frozen scientific settings changed: '+str(changes))
        if p['methods']!=old['methods']: raise ValueError('Method coverage changed')
        if len(p['datasets'])!=3 or len(p['initialization_seeds'])!=3:
            raise ValueError('C1 requires exactly three datasets and three initializations')
        if set(p['initialization_seeds']) & {701,709,719,727,733,739}:
            raise ValueError('Previous PI4 initialization reused')
    ids=[d['id'] for d in p['datasets']]
    if len(ids)!=len(set(ids)) or any('/' in d or '..' in d for d in ids):
        raise ValueError('Invalid dataset ids')
    ranges=[]
    for d in p['datasets']:
        # make() uses a subset of offsets 0..205; reserve the whole interval.
        x=set(range(d['data_seed'],d['data_seed']+206))
        if p['stage']=='gpu' and (x & set(range(920407,920613)) or x & set(range(920509,920715))):
            raise ValueError('Old PI4 generator seed range overlaps')
        if any(x&y for y in ranges): raise ValueError('Dataset generator streams overlap')
        ranges.append(x)
    if len(set(p['initialization_seeds']))!=len(p['initialization_seeds']):
        raise ValueError('Duplicate initialization')
def config_for(p: dict,d: dict)->dict:
    c=dict(p['fit']);c.update(protocol=p['protocol'],data_seed=d['data_seed'],
       seeds=p['initialization_seeds'],methods=p['methods'])
    return c

def jobs(p:dict)->list[dict]:
    # Randomized order fixed before outcomes, stratified by initialization block.
    rng=random.Random(p['schedule_seed']);out=[]
    for seed in p['initialization_seeds']:
        block=[{'dataset':d['id'],'method':m,'seed':seed,
                'trial_id':f'{m}_s{seed}'} for d in p['datasets'] for m in p['methods']]
        rng.shuffle(block);out.extend(block)
    return out

def source_manifest()->dict:
    fs=[]
    for folder in ('pi2','pi4','c1','configs','scripts','docs'):
        fs.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ('AGENTS.md','CODEX_START_CN.md','README_CN.md'):
        if (ROOT/name).exists(): fs.append(ROOT/name)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(fs)}

def verify_core()->dict:
    frozen=read(ROOT/'provenance/ORIGINAL_CORE_SHA256.json')
    bad=[k for k,h in frozen.items() if not (ROOT/k).exists() or sha(ROOT/k)!=h]
    return {'files':len(frozen),'mismatches':bad,'status':'PASS' if not bad else 'FAIL'}

def prerequisites(run:Path,p:dict)->list[dict]:
    bad=[]
    for j in jobs(p):
        td=run/'datasets'/j['dataset']/'trials'/j['trial_id']
        try:
            parent=read(td/'parent.json');worker=read(td/'result.json')
            ck=td/'best.pt'
            ok=(parent['status']=='COMPLETE' and parent['exit_code']==0 and
                parent['elapsed_seconds']<=p['fit']['process_limit_seconds'] and
                worker['status']=='COMPLETE' and worker['updates']==p['fit']['updates'] and
                ck.exists() and sha(ck)==worker['checkpoint_sha256'])
            why='incomplete worker/parent/update count/checkpoint hash'
        except (OSError,ValueError,KeyError) as e: ok=False;why=repr(e)
        if not ok: bad.append({**j,'reason':why})
    return bad

def release(run:Path,p:dict)->None:
    lock=run/'TEST_RELEASE_LOCK.json'
    if lock.exists(): raise RuntimeError('Test release already occurred; no repeat selection')
    bad=prerequisites(run,p)
    if bad: raise RuntimeError(f'{len(bad)} missing or invalid fits; no test release')
    original=read(run/'SOURCE_MANIFEST.json')
    if original!=source_manifest():raise RuntimeError('Source changed after launch')
    write(lock,{'status':'RELEASED_ONCE','training_closed':True,
        'scope':'All datasets trained and checkpoint-frozen before any test scoring',
        'protocol':p['protocol']})
