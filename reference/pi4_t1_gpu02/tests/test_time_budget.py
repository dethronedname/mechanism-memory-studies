from __future__ import annotations
import copy,json,sys,time,subprocess
from pathlib import Path
import pytest
from tc1.budget import choose_checkpoint,summarize_budgets,next_slot,admission,fit_complete
from tc1.protocol import ROOT,read,validate,jobs,config_for,verify_frozen,check_trial
from tc1.runner import spawn,write


def row(i,ready,score):
    return dict(step=i,selection_score=score,validation_started_elapsed=ready-.3,validation_complete_elapsed=ready-.2,
      checkpoint_saved_elapsed=ready-.1,checkpoint_ready_elapsed=ready,checkpoint_path=f'checkpoints/{i}.pt',checkpoint_sha256='abc')


def test_54_jobs_and_balanced_blocks():
    p=read(ROOT/'configs/time_budget.json');validate(p);js=jobs(p)
    assert len(js)==54 and len({(j['dataset'],j['trial_id']) for j in js})==54
    for i in range(0,54,6):
        block=js[i:i+6];assert len({(j['dataset'],j['seed']) for j in block})==1
        assert {j['method'] for j in block}==set(p['methods'])
    assert jobs(p)==js


def test_no_new_search_b1_lock_inherited():
    p=read(ROOT/'configs/time_budget.json')
    lock=read(ROOT/'provenance/B1_HYPERPARAMETER_LOCK.json')
    assert p['learning_rates']==lock['learning_rates']
    assert all(j['phase']=='time_comparison' for j in jobs(p))
    assert 'development' not in p and 'updates' not in p['fit']


def test_seventeen_model_and_baseline_files_unchanged():
    assert verify_frozen()=={'status':'PASS','files':17,'mismatches':[]}


def test_started_before_budget_does_not_make_late_save_eligible():
    a=row(1,149.,1.);b=row(2,150.1,.1);b['validation_started_elapsed']=149.5;b['validation_complete_elapsed']=149.6;b['checkpoint_saved_elapsed']=150.
    assert choose_checkpoint([a,b],150)['step']==1
    assert choose_checkpoint([a,b],300)['step']==2


def test_save_not_just_validation_must_be_finished():
    a=row(1,140.,.8);b=row(2,155.,.1);b['validation_started_elapsed']=130.;b['validation_complete_elapsed']=140.
    assert choose_checkpoint([a,b],150)['step']==1


def test_boundary_inclusive_and_missing_budget():
    a=row(1,150.,.3)
    assert choose_checkpoint([a],150)==a
    assert choose_checkpoint([a],149.9) is None
    assert summarize_budgets([a],[100,150])['100']['status']=='NOT_AVAILABLE_WITHIN_BUDGET'


def test_best_validation_not_latest_or_test():
    curve=[row(1,30,.6),row(2,60,.3),row(3,90,.5)]
    for i,r in enumerate(curve):r['test_mse']=3-i
    assert choose_checkpoint(curve,100)['step']==2
    assert choose_checkpoint(curve,45)['step']==1


def test_deterministic_tie_uses_earlier_saved():
    assert choose_checkpoint([row(2,50,.3),row(1,30,.3)],100)['step']==1


@pytest.mark.parametrize('change',[{'checkpoint_ready_elapsed':float('inf')},{'selection_score':float('nan')},
  {'checkpoint_saved_elapsed':1.},{'checkpoint_path':'../../escape.pt'},{'checkpoint_sha256':''}])
def test_impossible_or_nonfinite_rows_rejected(change):
    a=row(1,50,.3);a.update(change)
    with pytest.raises(ValueError):choose_checkpoint([a],100)


def test_duplicate_checkpoint_rejected():
    a=row(1,50,.3)
    with pytest.raises(ValueError):choose_checkpoint([a,a],100)


def test_wall_schedule_never_catches_up_with_duplicate_validation():
    assert next_slot(95,30)==120
    assert next_slot(30,30)==60
    assert not admission(570,600,30)
    assert not admission(568,600,30,recent_step=1.,recent_validation=2.)
    assert admission(400,600,30,recent_step=.3,recent_validation=.5)


def test_worker_complete_cannot_override_parent_timeout():
    pa={'status':'TIMEOUT','exit_code':0,'elapsed_seconds':600.05}
    w={'status':'COMPLETE','stop_reason':'TIME_BUDGET','updates':5000,'worker_elapsed':570}
    assert not fit_complete(pa,w,600,'gpu')
    pa.update(status='COMPLETE',elapsed_seconds=575)
    assert fit_complete(pa,w,600,'gpu')
    w['stop_reason']='EMERGENCY_UPDATE_CAP';assert not fit_complete(pa,w,600,'gpu')
    w['stop_reason']='CPU_SMOKE_UPDATE_CAP';assert not fit_complete(pa,w,600,'gpu')
    assert fit_complete(pa,w,600,'cpu')


@pytest.mark.parametrize('key,val',[('updates',1600),('validate_every',100)])
def test_no_update_count_formal_budget(key,val):
    p=read(ROOT/'configs/time_budget.json');p['fit'][key]=val
    with pytest.raises(ValueError):validate(p)


def test_smoke_cap_forbidden_on_gpu():
    p=read(ROOT/'configs/time_budget.json');p['time_policy']['cpu_smoke_max_updates']=2
    with pytest.raises(ValueError):validate(p)


def test_protocol_rejects_model_or_recipe_tweak():
    p=read(ROOT/'configs/time_budget.json');p['fit']['history_width']=32
    with pytest.raises(ValueError):validate(p)
    p=read(ROOT/'configs/time_budget.json');p['learning_rates']['fixed_behavior']=.001
    with pytest.raises(ValueError):validate(p)


def test_spawn_uses_exact_parent_origin_and_full_process_time(tmp_path):
    script=tmp_path/'child.py';capture=tmp_path/'stamp.json'
    script.write_text('import sys,json,time\na=float(sys.argv[-1]);time.sleep(.03)\njson.dump({"origin":a,"elapsed":time.monotonic()-a},open(sys.argv[1],"w"))\n')
    q=spawn([sys.executable,str(script),str(capture)],tmp_path/'log',2.,pass_clock=True)
    saved=json.loads(capture.read_text());assert q['status']=='COMPLETE'
    assert saved['origin']==q['parent_origin_monotonic']
    assert .03<saved['elapsed']<=q['elapsed_seconds']<=2.


def test_print_complete_then_hang_still_times_out(tmp_path):
    q=spawn([sys.executable,'-c','import time;print("COMPLETE",flush=True);time.sleep(2)'],tmp_path/'hang.log',.15)
    assert q['status']=='TIMEOUT' and q['timeout_observed']


def test_no_gpu_returns54_and_no_test_release(tmp_path):
    import torch
    if torch.cuda.is_available():pytest.skip('No-CUDA guard exercised only without CUDA')
    from tc1.runner import run_stage,collect
    path=tmp_path/'nogpu';code=run_stage('gpu','cuda:0',path)
    assert code==2 and len(read(path/'UNRUN.json'))==54
    assert not (path/'TEST_RELEASE_LOCK.json').exists()
    assert read(path/'EXECUTION_RECEIPT.json')['gpu']=='NOT_RUN'
    z=collect(path,tmp_path/'nogpu.zip');assert not z['mismatches']


def test_missing_fits_stop_before_release(tmp_path):
    from tc1.protocol import prerequisites
    p=read(ROOT/'configs/time_budget.json')
    assert len(prerequisites(tmp_path,p))==54
    assert not (tmp_path/'TEST_RELEASE_LOCK.json').exists()


def test_released_protocol_refuses_train_before_loading_data(tmp_path):
    import torch
    from tc1.experiment import fit
    root=tmp_path/'datasets'/'x';data=root/'data';data.mkdir(parents=True)
    write(root/'TEST_RELEASE_LOCK.json',{'status':'RELEASED'})
    with pytest.raises(RuntimeError,match='Training closed'):
        fit({},'fixed_behavior',1,data,root/'trials'/'a','cpu',time.monotonic())


def test_600_is_only_test_budget():
    p=read(ROOT/'configs/time_budget.json')
    assert p['time_policy']['primary_seconds']==600
    assert p['time_policy']['secondary_seconds']==[150,300]
    assert p['fit']['cleanup_reserve_seconds']==30
    assert p['resource']['total_limit_seconds']==16*3600
