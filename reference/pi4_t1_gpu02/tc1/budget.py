"""Budget and selection primitives. No tensors, test labels, or GPU are used here."""
from __future__ import annotations
import math
from pathlib import Path


def choose_checkpoint(curve: list[dict], budget: float) -> dict | None:
    """Best validation score whose validated, hashed checkpoint existed by budget.

    Validation starting before the deadline is insufficient. Weight serialization,
    verification and the validation result must both finish within the budget.
    Rows that cross a boundary remain evidence, but are never eligible there.
    """
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('budget must be positive and finite')
    eligible=[];seen=set()
    for row in curve:
        key=row['checkpoint_path']
        if key in seen:
            raise ValueError('duplicate checkpoint path')
        seen.add(key)
        p=Path(key)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('unsafe checkpoint path')
        times=[float(row[k]) for k in ('validation_started_elapsed','validation_complete_elapsed',
                                      'checkpoint_saved_elapsed','checkpoint_ready_elapsed')]
        score=float(row['selection_score'])
        if not all(math.isfinite(x) for x in times+[score]) or times[0]<0 or any(b<a for a,b in zip(times,times[1:])):
            raise ValueError('nonfinite or impossible checkpoint timestamps')
        if not row.get('checkpoint_sha256'):
            raise ValueError('missing checkpoint digest')
        if times[-1]<=budget:
            eligible.append(row)
    return min(eligible,key=lambda r:(r['selection_score'],r['checkpoint_ready_elapsed'],r['step'])) if eligible else None


def summarize_budgets(curve: list[dict], budgets: list[float]) -> dict:
    result={}
    for b in budgets:
        row=choose_checkpoint(curve,b)
        result[str(b)]={'status':'AVAILABLE' if row else 'NOT_AVAILABLE_WITHIN_BUDGET',
                        'budget_seconds':b,'selected':row}
    return result


def next_slot(elapsed: float, interval: float) -> float:
    if interval<=0 or elapsed<0:
        raise ValueError('invalid wall schedule')
    return (math.floor(elapsed/interval)+1)*interval


def admission(elapsed: float, limit: float, reserve: float,
              recent_step: float=0., recent_validation: float=0.) -> bool:
    """Do not start a known-cost update close to the soft stop. Not a guarantee."""
    if limit<=reserve or reserve<0:
        raise ValueError('invalid reserve')
    estimated=max(recent_step,0.)*2+max(recent_validation,0.)*1.5
    return elapsed+estimated < limit-reserve


def fit_complete(parent: dict, worker: dict, primary: float, stage: str) -> bool:
    good_reason={'TIME_BUDGET'} if stage=='gpu' else {'CPU_SMOKE_UPDATE_CAP','TIME_BUDGET'}
    return (parent.get('status')=='COMPLETE' and parent.get('exit_code')==0
            and parent.get('elapsed_seconds',float('inf'))<=primary
            and worker.get('status')=='COMPLETE' and worker.get('stop_reason') in good_reason
            and worker.get('updates',0)>0 and worker.get('worker_elapsed',float('inf'))<=primary)
