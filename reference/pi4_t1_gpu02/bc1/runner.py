from __future__ import annotations
import os,sys,time,platform,signal,subprocess,traceback,zipfile,hashlib,json
from pathlib import Path
from .protocol import (ROOT,read,validate,source_manifest,verify_core,development_jobs,confirmation_jobs,
    config_for,prerequisites,select_development)
from pi4.io import write,sha

class ResourceStop(RuntimeError):pass

def spawn(cmd,log,limit):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with log.open('w',encoding='utf8') as handle:
        proc=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        status='COMPLETE'
        try:code=proc.wait(timeout=max(.01,limit));status='COMPLETE' if code==0 else 'ERROR'
        except subprocess.TimeoutExpired:
            status='TIMEOUT'
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:code=proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                code=proc.wait(timeout=3)
    return {'status':status,'exit_code':code,'elapsed_seconds':time.monotonic()-start,'hard_limit_seconds':limit}

def task(action,run,device,dataset=None,trial=None,started=None):
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run=Path(run);p=read(run/'protocol.json')
    if read(run/'SOURCE_MANIFEST.json')!=source_manifest():raise RuntimeError('Source/protocol changed after run start')
    if verify_core()['status']!='PASS':raise RuntimeError('Original model/loss changed')
    if action=='prepare':
        from pi4.data import make
        dev=p['development'];root=run/'development';root.mkdir();cfg=config_for(p,dev,dev=True)
        write(root/'config.json',cfg);make(cfg,root/'data')
        for ds in p['datasets']:
            root=run/'datasets'/ds['id'];root.mkdir(parents=True);cfg=config_for(p,ds)
            write(root/'config.json',cfg);make(cfg,root/'data')
        from .models import model_for
        counts={}
        for m in p['methods']:
            torch.manual_seed(911);model=model_for(m)
            counts[m]={'parameters':sum(x.numel() for x in model.parameters()),'persistent_scalars_per_node':model.persistent_dimension,
                      'dynamic_dimension':model.d,'fixed_context_dimension':model.cdim,'model_config':model.config()}
        write(run/'MODEL_CENSUS.json',counts);return
    if action=='preflight':
        from .numerics import preflight
        cfg=read(run/'development/config.json');allchecks=[]
        # Verify both actual new optimizer LRs, original pair once: 30 triplets.
        for lr in p['development']['learning_rates']:
            cc=dict(cfg,learning_rate=lr,methods=p['methods'] if lr==.003 else p['development']['methods'])
            file=run/f'preflight_lr_{lr}.json'
            r=preflight(cc,run/'development/data',device,file)
            allchecks.extend(dict(x,learning_rate=lr) for x in r['checks'])
        write(run/'preflight.json',{'status':'PASS','checks':allchecks,'device':device,'no_cuda_fallback':True});return
    if action=='select':select_development(run,p);return
    if action=='finalize':
        from .analysis import finalize
        finalize(run,p);return
    if action=='fit':
        if (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('Training permanently closed')
        jobs=read(run/'TRIAL_MANIFEST.json');matches=[j for j in jobs if j['dataset']==dataset and j['trial_id']==trial]
        if len(matches)!=1:raise RuntimeError('Unplanned fit')
        j=matches[0]
        root=run/'development' if j['phase']=='development' else run/'datasets'/dataset
        cfg=read(root/'config.json')
        cfg['learning_rate']=j['learning_rate'] if j['phase']=='development' else read(run/'HYPERPARAMETER_LOCK.json')['learning_rates'][j['method']]
        td=root/'trials'/trial
        if (td/'result.json').exists() or (td/'best.pt').exists():raise RuntimeError('No retry/resume')
        write(td/'ACTUAL_FIT_CONFIG.json',cfg)
        from .experiment import fit
        fit(cfg,j['method'],j['seed'],root/'data',td,device,started=started);return
    if dataset not in [d['id'] for d in p['datasets']]:raise ValueError('Unknown confirm dataset')
    root=run/'datasets'/dataset;cfg=read(root/'config.json')
    if action=='evaluate':
        if not (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('No global release')
        from .experiment import evaluate
        out=evaluate(cfg,root,device)
        if out.get('status')!='EXPLORATORY_COMPLETE':raise RuntimeError('Evaluation prerequisites failed')
        return
    if action=='audit':
        from .numerics import audit_dataset
        audit_dataset(root,p['numerics'],ROOT,cfg);return
    raise ValueError(action)

def run_stage(stage,device,out):
    import torch,numpy,scipy
    start=time.monotonic();out=Path(out).resolve()
    if out.exists():raise FileExistsError('New output directory required; no overwrite')
    out.mkdir(parents=True)
    p=read(ROOT/'configs'/('challenge.json' if stage=='gpu' else 'smoke.json'));validate(p)
    jobs=development_jobs(p)+confirmation_jobs(p)
    write(out/'protocol.json',p);write(out/'TRIAL_MANIFEST.json',jobs);write(out/'SOURCE_MANIFEST.json',source_manifest())
    environment={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':numpy.__version__,
      'scipy':scipy.__version__,'cuda_available':torch.cuda.is_available(),'requested_device':device,
      'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES')}
    write(out/'environment.json',environment)
    if verify_core()['status']!='PASS':
        write(out/'DECISION.json',{'status':'INCONCLUSIVE','reason':'Frozen core mismatch'});return 3
    if stage=='gpu' and (device!='cuda:0' or not torch.cuda.is_available()):
        write(out/'DECISION.json',{'status':'NOT_RUN','gpu':'NOT_RUN','reason':'CUDA required, no CPU fallback'})
        write(out/'UNRUN.json',jobs);write(out/'EXECUTION_RECEIPT.json',{'status':'NOT_RUN','gpu':'NOT_RUN','planned_fits':len(jobs),'attempted_fits':0});return 2
    if stage=='cpu' and device!='cpu':raise ValueError('CPU smoke must use CPU')
    deadline=start+p['resource']['total_limit_seconds'];events=[];attempted=[];code=0
    base=[sys.executable,str(ROOT/'scripts/run.py'),'_task','--run',str(out),'--device',device]
    def stage_task(action,limit,dataset=None):
        remain=deadline-time.monotonic()-3
        if remain<=0:raise ResourceStop('Global hard cap')
        cmd=base+['--action',action]
        if dataset:cmd+=['--dataset',dataset]
        ev=spawn(cmd,out/'logs'/f'{action}_{dataset or "all"}.log',min(limit,remain))
        events.append(dict(ev,action=action,dataset=dataset));write(out/'STAGE_EVENTS.json',events)
        if ev['status']!='COMPLETE':raise RuntimeError(f'{action} failed; raw logs retained')
    def fit_job(j):
        if deadline-time.monotonic()<p['resource']['postprocess_reserve_seconds']+p['fit']['process_limit_seconds']:
            raise ResourceStop('Postprocessing reserve: no new fit, no automatic extension')
        td=(out/'development' if j['phase']=='development' else out/'datasets'/j['dataset'])/'trials'/j['trial_id'];td.mkdir(parents=True)
        t=time.monotonic();cmd=base+['--action','fit','--dataset',j['dataset'],'--trial',j['trial_id'],'--started',str(t)]
        pr=spawn(cmd,td/'worker.log',p['fit']['process_limit_seconds'])
        if pr['status']=='COMPLETE':
            try:
                wr=read(td/'result.json');pr['status']=wr['status']
                if wr['updates']!=p['fit']['updates']:pr['status']='BUDGET_STOP'
            except Exception as e:pr.update(status='ERROR',reason=repr(e))
        write(td/'parent.json',pr);attempted.append(j);print(j['dataset'],j['trial_id'],pr['status'],flush=True)
    try:
        stage_task('prepare',p['resource']['prepare_seconds'])
        if stage=='gpu':stage_task('preflight',p['resource']['preflight_seconds'])
        for j in development_jobs(p):fit_job(j)
        stage_task('select',120)
        for j in confirmation_jobs(p):fit_job(j)
        bad=prerequisites(out,p)
        if bad:
            write(out/'DECISION.json',{'status':'INCONCLUSIVE','reason':'Missing comparisons; no tests released','missing':bad});code=3
        else:
            write(out/'TEST_RELEASE_LOCK.json',{'status':'RELEASED_ONCE','unix_time':time.time(),'scope':'B1 fresh data only; old C1 untouched','hyperparameter_lock_sha256':sha(out/'HYPERPARAMETER_LOCK.json')})
            for ds in p['datasets']:stage_task('evaluate',p['resource']['evaluate_seconds_per_dataset'],ds['id'])
            for ds in p['datasets']:stage_task('audit',p['resource']['audit_seconds_per_dataset'],ds['id'])
            stage_task('finalize',p['resource']['finalize_seconds'])
            if read(out/'DECISION.json')['status'] in ('INCONCLUSIVE','CPU_ENGINEERING_FAIL'):code=3
    except ResourceStop as e:
        write(out/'DECISION.json',{'status':'INCONCLUSIVE','execution':'PARTIAL','reason':str(e)});code=3
    except Exception as e:
        write(out/'DECISION.json',{'status':'INCONCLUSIVE','reason':repr(e),'traceback':traceback.format_exc()});code=3
    finally:
        write(out/'UNRUN.json',[j for j in jobs if j not in attempted]);write(out/'WALL_COST.json',{'seconds':time.monotonic()-start,'limit_seconds':p['resource']['total_limit_seconds'],
            'includes':['development search','preflight','all fits including failures','evaluation','CPU full replay'], 'excludes':['implementation','packaging'],'not_gpu_kernel_time':True})
        verdict=read(out/'DECISION.json') if (out/'DECISION.json').exists() else {'status':'INCONCLUSIVE'}
        write(out/'EXECUTION_RECEIPT.json',{'status':verdict['status'],'attempted_fits':len(attempted),'planned_fits':len(jobs),'gpu':'EXECUTED' if stage=='gpu' else 'NOT_RUN','automatic_next_experiment':False})
        if not (out/'REPORT_CN.md').exists():(out/'REPORT_CN.md').write_text('# B1 执行结果\n\n'+json.dumps(verdict,ensure_ascii=False,indent=2),encoding='utf8')
    return code

def verify_package():
    path=ROOT/'MANIFEST_SHA256.json'
    if not path.exists():return {'status':'FAIL','reason':'Missing package manifest'}
    m=read(path);bad=[k for k,h in m.items() if not (ROOT/k).is_file() or sha(ROOT/k)!=h]
    core=verify_core();return {'status':'PASS' if not bad and core['status']=='PASS' else 'FAIL','files':len(m),'mismatches':bad,'original_core':core}

def collect(run,out):
    run=Path(run).resolve();out=Path(out).resolve()
    if out.exists() or run in out.parents:raise ValueError('New archive outside run required')
    if not (run/'protocol.json').exists():raise ValueError('Not a B1 result directory')
    out.parent.mkdir(parents=True,exist_ok=True);mf={};start=time.monotonic()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            rel=path.relative_to(run);keep=not ('data' in rel.parts and path.suffix=='.npz')
            mf['run/'+str(rel)]={'bytes':path.stat().st_size,'sha256':sha(path),'included':keep}
            if keep:z.write(path,'run/'+str(rel))
        for rel in source_manifest():
            path=ROOT/rel;z.write(path,'source/'+rel);mf['source/'+rel]={'bytes':path.stat().st_size,'sha256':sha(path),'included':True}
        for rel in ('provenance/ORIGINAL_CORE_SHA256.json','MANIFEST_SHA256.json'):
            path=ROOT/rel
            if path.exists():z.write(path,'source/'+rel);mf['source/'+rel]={'bytes':path.stat().st_size,'sha256':sha(path),'included':True}
        z.writestr('RETURN_MANIFEST.json',json.dumps(mf,ensure_ascii=False,indent=2))
        z.writestr('REGENERATION_CN.md','只排除可由pi4.data.make和各dataset/config.json重建的数据NPZ；全部权重、预测、学习曲线、失败和LR开发记录保留。绝不重训或重新选模。原始数据内容哈希在data/MANIFEST.json。\n')
    with zipfile.ZipFile(out) as z:
        bad=[k for k,v in mf.items() if v['included'] and hashlib.sha256(z.read(k)).hexdigest()!=v['sha256']];crc=z.testzip()
    rec={'file':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'included_files':sum(v['included'] for v in mf.values()),'mismatches':bad,'crc_error':crc,'seconds':time.monotonic()-start}
    write(str(out)+'.validation.json',rec)
    if bad or crc:raise RuntimeError('Return ZIP integrity failed')
    return rec
