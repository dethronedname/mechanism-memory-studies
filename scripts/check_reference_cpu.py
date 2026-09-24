#!/usr/bin/env python3
"""Run an explicit bounded CPU unit-test allowlist in isolated stage processes."""
from pathlib import Path
import argparse, importlib.util, json, os, subprocess, sys, tempfile
from verify_reference import ROOT, verify
STUDIES={
 'pi4':('pi4',{'tests/test_study.py':[
   'test_reset_matching_and_null','test_pair_loss_identity','test_no_future_target_leak_and_support_permutation',
   'test_fixed_context_remains_exact','test_behavior_zero_weights_matches_mse','test_same_model_node_permutation','test_model_reload']}),
 'c1':('pi4_c1',{'tests/test_confirmation.py':[
   'test_frozen_core_exact','test_configuration_frozen','test_schedule_complete_and_repeatable','test_gate_positive_and_negative',
   'test_legacy_element_failure_reported_without_abort','test_independent_scoring_and_null_denominator',
   'test_saved_difference_matches_frozen_float32_subtraction'],
   'tests/test_study.py':['test_fixed_context_remains_exact','test_no_future_target_leak_and_support_permutation']}),
 'b1':('pi4_b1',{'tests/test_baselines.py':[
   'test_all_real_objective_paths','test_context_persists_and_affects_every_step','test_support_order_and_node_permutation',
   'test_no_target_access_or_teacher_forcing','test_save_load','test_persistent_memory_definition',
   'test_candidate_and_old_control_identical','test_frozen_schedule']}),
 't1':('pi4_t1_gpu02',{'tests/test_time_budget.py':[
   'test_54_jobs_and_balanced_blocks','test_no_new_search_b1_lock_inherited','test_seventeen_model_and_baseline_files_unchanged',
   'test_started_before_budget_does_not_make_late_save_eligible','test_save_not_just_validation_must_be_finished',
   'test_best_validation_not_latest_or_test','test_deterministic_tie_uses_earlier_saved',
   'test_worker_complete_cannot_override_parent_timeout','test_no_update_count_formal_budget',
   'test_protocol_rejects_model_or_recipe_tweak','test_600_is_only_test_budget']})
}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',choices=['all',*STUDIES],default='all')
    a=p.parse_args();verify(study=a.study)
    missing=[n for n in ('torch','numpy','scipy','pytest') if importlib.util.find_spec(n) is None]
    if missing:p.error('Use an existing CPU-capable environment with: '+', '.join(missing))
    results=[]
    for study,(folder,selection) in STUDIES.items():
        if a.study not in ('all',study):continue
        cwd=ROOT/'reference'/folder
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                 PYTEST_ADDOPTS='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env['PYTHONPATH']=str(cwd)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        with tempfile.TemporaryDirectory(prefix='mechanism-memory-cpu-') as td:
            nodes=[file+'::'+name for file,names in selection.items() for name in names]
            command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--basetemp='+td]+nodes
            result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=90)
        print('STUDY '+study+'\n'+result.stdout+result.stderr,flush=True)
        results.append(dict(study=study,returncode=result.returncode,selected_functions=len(nodes)))
    verify(study=a.study)
    print(json.dumps(dict(status='PASS' if all(x['returncode']==0 for x in results) else 'FAIL',
                         scope='CPU unit checks only; no fit/stage launcher, historical weight replay or GPU test',studies=results),indent=2))
    return int(any(x['returncode'] for x in results))
if __name__=='__main__':raise SystemExit(main())
