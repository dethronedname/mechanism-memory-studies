from __future__ import annotations
import os,sys,time,platform,signal,subprocess,traceback,zipfile,hashlib,json
from pathlib import Path
from .protocol import ROOT,read,sha,validate,source_manifest,verify_frozen,jobs,config_for,prerequisites,check_trial

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,path)

class ResourceStop(RuntimeError):pass

def spawn(cmd,log,limit,pass_clock=False):
    """Single origin for child budget AND parent deadline. Timeout never relabeled."""
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as f:
        start=time.monotonic();deadline=start+limit
        command=list(cmd)+(['--started',repr(start)] if pass_clock else [])
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        status='COMPLETE';code=None;timedout=False
        try:
            code=proc.wait(timeout=max(.001,deadline-time.monotonic()))
            exit_observed=time.monotonic();status='COMPLETE' if code==0 else 'ERROR'
            if exit_observed>deadline:status='TIMEOUT';timedout=True
        except subprocess.TimeoutExpired:
            timedout=True;status='TIMEOUT'
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:code=proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                code=proc.wait(timeout=3)
        end=time.monotonic()
    return dict(status=status,exit_code=code,elapsed_seconds=end-start,hard_limit_seconds=limit,
                parent_origin_monotonic=start,parent_end_monotonic=end,deadline_monotonic=deadline,
                timeout_observed=timedout,cleanup_overrun_seconds=max(0.,end-deadline),
                note='Deadline includes subprocess startup. Reaping after a timeout is charged, not training allowance.')


def task(action,run,device,dataset=None,trial=None,started=None):
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run=Path(run);p=read(run/'protocol.json')
    source=read(run/'SOURCE_MANIFEST.json')
    if source!=source_manifest() or verify_frozen()['status']!='PASS':raise RuntimeError('Frozen source changed')
    expected=read(ROOT/'configs'/('time_budget.json' if p['stage']=='gpu' else 'time_smoke.json'))
    if p!=expected or read(run/'TRIAL_MANIFEST.json')!=jobs(p):raise RuntimeError('Run protocol/schedule mutated')
    if action=='prepare':
        from pi4.data import make
        from bc1.models import model_for
        for d in p['datasets']:
            folder=run/'datasets'/d['id'];folder.mkdir(parents=True,exist_ok=False)
            cfg=config_for(p,d);write(folder/'config.json',cfg);make(cfg,folder/'data')
        census={}
        for m in p['methods']:
            torch.manual_seed(911);model=model_for(m,p['fit']['width'],p['fit']['history_width'])
            census[m]=dict(parameters=sum(x.numel() for x in model.parameters()),persistent_scalars_per_node=model.persistent_dimension,
                           dynamic_dimension=model.d,fixed_context_dimension=model.cdim,model_config=model.config())
        write(run/'MODEL_CENSUS.json',census);return
    if action=='preflight':
        from .numerics import preflight
        first=run/'datasets'/p['datasets'][0]['id']
        cfg=read(first/'config.json')
        if len(set(p['learning_rates'].values()))!=1:raise ValueError('Mixed LR requires prespecified preflight')
        preflight(cfg,first/'data',device,run/'preflight.json');return
    if action=='finalize':
        from .analysis import finalize
        finalize(run,p);return
    if dataset not in [d['id'] for d in p['datasets']]:raise ValueError('Unplanned dataset')
    root=run/'datasets'/dataset;cfg=read(root/'config.json')
    d=next(d for d in p['datasets'] if d['id']==dataset)
    if cfg!=config_for(p,d):raise RuntimeError('Dataset fit config changed')
    if action=='fit':
        if (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('Training permanently closed')
        found=[j for j in jobs(p) if j['dataset']==dataset and j['trial_id']==trial]
        if len(found)!=1:raise RuntimeError('Unplanned fit')
        j=found[0];cfg=dict(cfg,learning_rate=j['learning_rate']);td=root/'trials'/trial
        write(td/'ACTUAL_FIT_CONFIG.json',cfg)
        from .experiment import fit
        r=fit(cfg,j['method'],j['seed'],root/'data',td,device,started)
        if r['status']!='COMPLETE':raise RuntimeError('Worker did not reach a valid budget stop')
        return
    if action=='evaluate':
        if not (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('No global test release')
        if prerequisites(run,p):raise RuntimeError('Budget prerequisites failed')
        from .evaluation import evaluate
        s=evaluate(cfg,root,device)
        if s.get('status')!='EXPLORATORY_COMPLETE':raise RuntimeError('Evaluation incomplete')
        return
    if action=='audit':
        from .numerics import audit_dataset
        audit_dataset(root,p['numerics'],ROOT,cfg);return
    raise ValueError('Unplanned action')


def run_stage(stage,device,out):
    import torch,numpy,scipy
    start=time.monotonic();out=Path(out).resolve()
    if out.exists():raise FileExistsError('New directory required; no overwrite or resume')
    out.mkdir(parents=True)
    p=read(ROOT/'configs'/('time_budget.json' if stage=='gpu' else 'time_smoke.json'));validate(p);plan=jobs(p)
    write(out/'protocol.json',p);write(out/'TRIAL_MANIFEST.json',plan);write(out/'SOURCE_MANIFEST.json',source_manifest())
    write(out/'HYPERPARAMETER_LOCK.json',dict(status='INHERITED_FROM_B1_DEV',learning_rates=p['learning_rates'],
        source='provenance/B1_HYPERPARAMETER_LOCK.json',source_sha256=sha(ROOT/'provenance/B1_HYPERPARAMETER_LOCK.json'),new_search_trials=0))
    env=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,
       requested_device=device,cuda_available=torch.cuda.is_available(),CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
       gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,clock=time.get_clock_info('monotonic')._asdict() if hasattr(time.get_clock_info('monotonic'),'_asdict') else str(time.get_clock_info('monotonic')))
    write(out/'environment.json',env)
    if stage=='gpu' and (device!='cuda:0' or not torch.cuda.is_available()):
        write(out/'DECISION.json',dict(status='NOT_RUN',gpu='NOT_RUN',reason='CUDA required; no CPU fallback'))
        write(out/'UNRUN.json',plan);write(out/'EXECUTION_RECEIPT.json',dict(status='NOT_RUN',gpu='NOT_RUN',planned_fits=len(plan),attempted_fits=0,automatic_next_experiment=False));return 2
    if stage=='cpu' and device!='cpu':raise ValueError('CPU smoke requires CPU')
    if verify_frozen()['status']!='PASS':
        write(out/'DECISION.json',dict(status='INCONCLUSIVE',reason='Model/loss/source mismatch'));write(out/'UNRUN.json',plan);return 3
    deadline=start+p['resource']['total_limit_seconds'];events=[];attempted=[];code=0
    base=[sys.executable,str(ROOT/'scripts/run_t1.py'),'_task','--run',str(out),'--device',device]
    def stage_task(action,limit,dataset=None):
        left=deadline-time.monotonic()-p['resource']['packaging_reserve_seconds']
        if left<=0:raise ResourceStop('Whole-stage budget exhausted; no auto extension')
        cmd=base+['--action',action]+(['--dataset',dataset] if dataset else [])
        ev=spawn(cmd,out/'logs'/f'{action}_{dataset or "all"}.log',min(left,limit))
        events.append(dict(ev,action=action,dataset=dataset));write(out/'STAGE_EVENTS.json',events)
        if ev['status']!='COMPLETE':raise RuntimeError(f'{action} failed; evidence retained')
    try:
        stage_task('prepare',p['resource']['prepare_seconds'])
        if stage=='gpu':stage_task('preflight',p['resource']['preflight_seconds'])
        for j in plan:
            if deadline-time.monotonic()<p['resource']['postprocess_reserve_seconds']+p['time_policy']['primary_seconds']+p['resource']['packaging_reserve_seconds']:
                raise ResourceStop('Postprocessing reserve reached; no new fit')
            td=out/'datasets'/j['dataset']/'trials'/j['trial_id'];td.mkdir(parents=True,exist_ok=False)
            cmd=base+['--action','fit','--dataset',j['dataset'],'--trial',j['trial_id']]
            pr=spawn(cmd,td/'worker.log',p['time_policy']['primary_seconds'],pass_clock=True)
            # Parent status is not replaced with worker COMPLETE; both independently checked.
            if pr['status']=='COMPLETE':
                try:
                    r=read(td/'result.json')
                    if r['status']!='COMPLETE':pr.update(status='ERROR',reason='worker incomplete')
                except Exception as e:pr.update(status='ERROR',reason=repr(e))
            write(td/'parent.json',pr);attempted.append(j)
            print(j['dataset'],j['trial_id'],pr['status'],round(pr['elapsed_seconds'],3),flush=True)
        bad=prerequisites(out,p)
        write(out/'FIT_BUDGET_AUDIT.json',dict(status='PASS' if not bad else 'FAIL',problems=bad,planned=len(plan)))
        if bad:
            write(out/'DECISION.json',dict(status='INCONCLUSIVE',reason='Missing or invalid fit; tests remain sealed',problems=bad));code=3
        else:
            write(out/'TEST_RELEASE_LOCK.json',dict(status='RELEASED_ONCE',scope='Only primary time budget; no early-budget test ranking',
                  time=time.time(),all_fit_count=len(plan),budget_seconds=p['time_policy']['primary_seconds'],hyperparameter_lock_sha256=sha(out/'HYPERPARAMETER_LOCK.json')))
            for ds in p['datasets']:stage_task('evaluate',p['resource']['evaluate_seconds_per_dataset'],ds['id'])
            for ds in p['datasets']:stage_task('audit',p['resource']['audit_seconds_per_dataset'],ds['id'])
            stage_task('finalize',p['resource']['finalize_seconds'])
            if read(out/'DECISION.json')['status'] in ('INCONCLUSIVE','CPU_ENGINEERING_FAIL'):code=3
    except ResourceStop as e:
        write(out/'DECISION.json',dict(status='INCONCLUSIVE',execution='PARTIAL',reason=str(e)));code=3
    except Exception as e:
        write(out/'DECISION.json',dict(status='INCONCLUSIVE',reason=repr(e),traceback=traceback.format_exc()));code=3
    finally:
        write(out/'UNRUN.json',[j for j in plan if j not in attempted])
        fit_costs=[]
        for j in attempted:
            td=out/'datasets'/j['dataset']/'trials'/j['trial_id'];pa=read(td/'parent.json')
            fit_costs.append(dict(j,**pa))
        elapsed=time.monotonic()-start
        write(out/'WALL_COST.json',dict(seconds=elapsed,limit_seconds=p['resource']['total_limit_seconds'],stage_start_monotonic=start,
           stage_deadline_monotonic=deadline,fits=fit_costs,stage_events=events,all_fit_seconds=sum(x['elapsed_seconds'] for x in fit_costs),
           same_fit_cap_not_identical_realized_seconds=True,new_search_trials=0,
           includes=['common data preparation','preflight','all fits and failures','primary test scoring','CPU replay'],
           packaging='separate returned manifest clock; budget remainder checked by collection',
           excludes=['prior B1 LR search; recorded in provenance','implementation work'],
           not_gpu_kernel_time=True))
        dec=read(out/'DECISION.json') if (out/'DECISION.json').exists() else dict(status='INCONCLUSIVE')
        write(out/'EXECUTION_RECEIPT.json',dict(status=dec['status'],planned_fits=len(plan),attempted_fits=len(attempted),
                  gpu='EXECUTED' if stage=='gpu' else 'NOT_RUN',automatic_next_experiment=False))
        if not (out/'REPORT_CN.md').exists():
            (out/'REPORT_CN.md').write_text('# PI-4 T1执行记录\n\n'+json.dumps(dec,ensure_ascii=False,indent=2),encoding='utf8')
    return code


def verify_package():
    path=ROOT/'MANIFEST_SHA256.json'
    if not path.exists():return dict(status='FAIL',reason='Package manifest missing')
    m=read(path);bad=[k for k,h in m.items() if not (ROOT/k).is_file() or sha(ROOT/k)!=h];core=verify_frozen()
    return dict(status='PASS' if not bad and core['status']=='PASS' else 'FAIL',files=len(m),mismatches=bad,frozen_b1=core)


def collect(run,out):
    run=Path(run).resolve();out=Path(out).resolve()
    if out.exists() or run in out.parents:raise ValueError('New archive outside run required')
    p=read(run/'protocol.json')
    if not p['protocol'].startswith('PI4-T1'):raise ValueError('Not a T1 run')
    out.parent.mkdir(parents=True,exist_ok=True);manifest={};t=time.monotonic()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            rel=path.relative_to(run);keep=not ('data' in rel.parts and path.suffix=='.npz');key='run/'+str(rel)
            manifest[key]=dict(bytes=path.stat().st_size,sha256=sha(path),included=keep)
            if keep:z.write(path,key)
        for rel in source_manifest():
            path=ROOT/rel;key='source/'+rel;z.write(path,key);manifest[key]=dict(bytes=path.stat().st_size,sha256=sha(path),included=True)
        z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('REGENERATION_CN.md','仅排除可用冻结pi4.data.make与各dataset/config.json重建的数据NPZ。保留全部检查点、更新日志、预算时钟、失败、预测与源码。禁止重新训练或测试选模。\n')
    with zipfile.ZipFile(out) as z:
        crc=z.testzip();bad=[k for k,v in manifest.items() if v['included'] and hashlib.sha256(z.read(k)).hexdigest()!=v['sha256']]
    wc=read(run/'WALL_COST.json') if (run/'WALL_COST.json').exists() else None
    budget_exceeded=bool(wc and time.monotonic()>wc['stage_deadline_monotonic'])
    result=dict(bytes=out.stat().st_size,sha256=sha(out),included_files=sum(v['included'] for v in manifest.values()),
        mismatches=bad,crc_error=crc,packaging_seconds=time.monotonic()-t,
        whole_stage_budget_exceeded_at_collection=budget_exceeded,
        budget_note='Archive is retained even on resource overrun; not a fresh training allowance.')
    write(str(out)+'.validation.json',result)
    if bad or crc:raise RuntimeError('Archive integrity failure')
    return result
