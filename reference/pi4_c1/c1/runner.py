"""Single-card, bounded confirmation. No scheduler access, retries, tuning or test-guided fitting."""
from __future__ import annotations
import os,sys,time,json,subprocess,signal,traceback,platform,zipfile
from pathlib import Path
from .protocol import ROOT,read,validate,config_for,jobs,source_manifest,verify_core,prerequisites,release
from pi4.io import write,sha

class ResourceStop(RuntimeError):pass

def spawn(command:list[str],log:Path,limit:float)->dict:
    """Kill this launched process group only; retain exit status and charge cleanup."""
    if limit<=0:raise ResourceStop('No remaining time')
    start=time.monotonic();log.parent.mkdir(parents=True,exist_ok=True)
    timed_out=False
    with log.open('w',encoding='utf8') as f:
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        try:code=proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:code=proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                code=proc.wait()
    elapsed=time.monotonic()-start
    return {'status':'TIMEOUT' if timed_out or elapsed>limit else 'COMPLETE' if code==0 else 'ERROR',
            'exit_code':code,'elapsed_seconds':elapsed,'hard_limit_seconds':limit,
            'counts_towards_budget':True,'command':command,'timeout_observed':timed_out}

def task(action:str,run:Path,dataset:str|None=None,method:str|None=None,seed:int|None=None,device='cpu',started=None)->None:
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=read(run/'protocol.json');validate(p)
    if p['stage']=='gpu' and action in ('fit','evaluate','preflight') and (device!='cuda:0' or not torch.cuda.is_available()):
        raise RuntimeError('Formal task requires assigned CUDA device; no fallback')
    if action=='prepare':
        from pi4.data import make
        for d in p['datasets']:
            path=run/'datasets'/d['id'];path.mkdir(parents=True,exist_ok=True)
            c=config_for(p,d);write(path/'config.json',c);make(c,path/'data')
        return
    if action=='preflight':
        from .numerics import preflight
        first=run/'datasets'/p['datasets'][0]['id']
        preflight(read(first/'config.json'),first/'data',device,run/'preflight.json');return
    if action=='finalize':
        from .analysis import aggregate
        aggregate(run,p);return
    path=run/'datasets'/str(dataset)
    if dataset not in [d['id'] for d in p['datasets']]:raise ValueError('Unknown dataset')
    c=read(path/'config.json')
    if action=='fit':
        if (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('All training closed by global test release')
        if method not in p['methods'] or seed not in p['initialization_seeds']:raise ValueError('Unplanned fit')
        if read(run/'SOURCE_MANIFEST.json')!=source_manifest():raise RuntimeError('Frozen source changed')
        td=path/'trials'/f'{method}_s{seed}'
        if (td/'result.json').exists() or (td/'best.pt').exists():raise RuntimeError('No retry/resume/overwrite')
        from pi4.experiment import fit
        fit(c,method,seed,path/'data',td,device,started=started);return
    if action=='evaluate':
        if not (run/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('No global test release')
        from pi4.experiment import evaluate
        out=evaluate(c,path,device)
        if out.get('status')!='EXPLORATORY_COMPLETE':raise RuntimeError('Incomplete evaluation')
        return
    if action=='audit':
        from .numerics import audit_dataset
        # Even an audit failure writes all available records; do not abort at first array.
        audit_dataset(path,p['numerics'],ROOT,c);return
    raise ValueError(action)

def run_stage(stage:str,device:str,out:Path)->int:
    import torch,numpy,scipy
    start=time.monotonic();out=out.resolve()
    if out.exists():raise FileExistsError('New run directory required; no overwrite')
    out.mkdir(parents=True)
    p=read(ROOT/'configs'/('confirm.json' if stage=='gpu' else 'smoke.json'));validate(p)
    write(out/'protocol.json',p);schedule=jobs(p);write(out/'TRIAL_MANIFEST.json',schedule)
    frozen=source_manifest();write(out/'SOURCE_MANIFEST.json',frozen)
    environment={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
       'numpy':numpy.__version__,'scipy':scipy.__version__,'cuda_available':torch.cuda.is_available(),
       'requested_device':device,'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES'),
       'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       'purpose':'Fresh-data task replication; no timing/kernel search'}
    write(out/'environment.json',environment);events=[];code=0
    if verify_core()['status']!='PASS':
        write(out/'DECISION.json',{'status':'INCONCLUSIVE','reason':'Original core modified'});return 3
    if stage=='gpu' and (device!='cuda:0' or not torch.cuda.is_available()):
        write(out/'DECISION.json',{'status':'NOT_RUN','gpu':'NOT_RUN','reason':'CUDA required; no CPU fallback'})
        write(out/'UNRUN.json',schedule);write(out/'EXECUTION_RECEIPT.json',{'status':'NOT_RUN','gpu':'NOT_RUN','attempted_fits':0})
        return 2
    if stage!='gpu' and device!='cpu':raise ValueError('CPU smoke must use CPU')
    deadline=start+p['resource']['total_limit_seconds']
    base=[sys.executable,str(ROOT/'scripts/run.py'),'_task','--run',str(out),'--device',device]
    def stage_task(action,limit,dataset=None):
        remain=deadline-time.monotonic()-3
        if remain<=0:raise ResourceStop('Overall hard limit')
        cmd=base+['--action',action]
        if dataset:cmd+=['--dataset',dataset]
        ev=spawn(cmd,out/'logs'/f'{action}_{dataset or "all"}.log',min(limit,remain))
        ev.update(action=action,dataset=dataset);events.append(ev);write(out/'STAGE_EVENTS.json',events)
        if ev['status']!='COMPLETE':raise RuntimeError(f'{action} {dataset}: {ev["status"]}')
    try:
        stage_task('prepare',p['resource']['prepare_seconds'])
        if stage=='gpu':stage_task('preflight',p['resource']['preflight_seconds'])
        # All 45 fits, not one dataset at a time followed by adaptive evaluation.
        for j in schedule:
            if deadline-time.monotonic()<p['resource']['postprocess_reserve_seconds']+p['fit']['process_limit_seconds']:
                raise ResourceStop('Reserve exhausted; no test release, no hidden continuation')
            td=out/'datasets'/j['dataset']/'trials'/j['trial_id'];td.mkdir(parents=True)
            t=time.monotonic()
            cmd=base+['--action','fit','--dataset',j['dataset'],'--method',j['method'],
                      '--seed',str(j['seed']),'--started',str(t)]
            pr=spawn(cmd,td/'worker.log',p['fit']['process_limit_seconds'])
            if pr['status']=='COMPLETE':
                try:
                    wr=read(td/'result.json')
                    pr['status']=wr['status']
                    if wr['updates']!=p['fit']['updates']:pr['status']='BUDGET_STOP'
                except Exception as e:pr.update(status='ERROR',reason=repr(e))
            write(td/'parent.json',pr);print(j,pr['status'],flush=True)
        bad=prerequisites(out,p)
        if bad:
            write(out/'DECISION.json',{'status':'INCONCLUSIVE','reason':'Missing required comparisons; tests not released','missing':bad});code=3
        else:
            release(out,p)
            for d in p['datasets']:stage_task('evaluate',p['resource']['evaluate_seconds_per_dataset'],d['id'])
            # CPU replay is included in the same total budget and written separately.
            for d in p['datasets']:stage_task('audit',p['resource']['audit_seconds_per_dataset'],d['id'])
            stage_task('finalize',p['resource']['finalize_seconds'])
            if read(out/'DECISION.json')['status'] in ('INCONCLUSIVE','CPU_ENGINEERING_FAIL'):code=3
    except ResourceStop as e:
        write(out/'DECISION.json',{'status':'INCONCLUSIVE','execution':'PARTIAL','reason':str(e)});code=3
    except Exception as e:
        write(out/'DECISION.json',{'status':'INCONCLUSIVE','reason':repr(e),'traceback':traceback.format_exc()});code=3
    finally:
        attempted=[j for j in schedule if (out/'datasets'/j['dataset']/'trials'/j['trial_id']/'parent.json').exists()]
        unrun=[j for j in schedule if j not in attempted];write(out/'UNRUN.json',unrun)
        write(out/'WALL_COST.json',{'total_seconds':time.monotonic()-start,
             'limit_seconds':p['resource']['total_limit_seconds'],
             'includes':['data generation','preflight','all fitting and failures','evaluation','full CPU audit','report'],
             'excludes':['software implementation','packaging after run'],
             'caveat':'Process termination may require a few seconds beyond deadline; never used for extra optimization.'})
        verdict=read(out/'DECISION.json') if (out/'DECISION.json').exists() else {'status':'INCONCLUSIVE'}
        write(out/'EXECUTION_RECEIPT.json',{'status':verdict['status'],'attempted_fits':len(attempted),
             'planned_fits':len(schedule),'unrun_fits':len(unrun),'gpu':'EXECUTED' if stage=='gpu' else 'NOT_RUN',
             'automatic_next_experiment':False})
        if not (out/'REPORT_CN.md').exists():
            (out/'REPORT_CN.md').write_text('# PI-4 C1 运行记录\n\n'+json.dumps(verdict,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return code

def verify_package()->dict:
    file=ROOT/'MANIFEST_SHA256.json'
    if not file.exists():return {'status':'FAIL','reason':'No delivery manifest'}
    m=read(file);bad=[k for k,h in m.items() if not (ROOT/k).is_file() or sha(ROOT/k)!=h]
    core=verify_core()
    return {'status':'PASS' if not bad and core['status']=='PASS' else 'FAIL','files':len(m),'mismatches':bad,'original_core':core}

def collect(run:Path,out:Path)->dict:
    run=run.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('Do not overwrite a return archive')
    if run==out or run in out.parents:raise ValueError('Return archive must be outside the run directory')
    if not (run/'protocol.json').exists():raise FileNotFoundError('Not a C1 run')
    out.parent.mkdir(parents=True,exist_ok=True);manifest={};excluded=[];start=time.monotonic()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            rel=path.relative_to(run)
            # Regenerable simulator inputs/labels are large; weights, every prediction and curves retained.
            include=not ('data' in rel.parts and path.suffix=='.npz')
            entry={'bytes':path.stat().st_size,'sha256':sha(path),'included':include}
            manifest['run/'+str(rel)]=entry
            if include:z.write(path,'run/'+str(rel))
            else:excluded.append(str(rel))
        for rel in source_manifest():
            path=ROOT/rel;z.write(path,'source/'+rel)
            manifest['source/'+rel]={'bytes':path.stat().st_size,'sha256':sha(path),'included':True}
        for name in ['provenance/ORIGINAL_CORE_SHA256.json','MANIFEST_SHA256.json']:
            path=ROOT/name
            if path.exists():
                z.write(path,'source/'+name);manifest['source/'+name]={'bytes':path.stat().st_size,'sha256':sha(path),'included':True}
        z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('REGENERATION_CN.md','未包含的文件只有可由冻结模拟器和配置重建的数据NPZ。先查看run/protocol.json及各datasets/*/config.json，调用pi4.data.make；不得重训或解锁后继续选模。数据内容哈希在data/MANIFEST.json，ZIP字节哈希不用于重新生成文件比较。所有小权重与预测均包含。\n')
    # Verify the finished archive, not just the input files.
    with zipfile.ZipFile(out) as z:
        bad=[]
        import hashlib
        for rel,item in manifest.items():
            if item['included'] and hashlib.sha256(z.read(rel)).hexdigest()!=item['sha256']:bad.append(rel)
        crc=z.testzip()
    result={'file':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'included_files':sum(v['included'] for v in manifest.values()),
            'excluded_data_files':len(excluded),'mismatches':bad,'crc_error':crc,'seconds':time.monotonic()-start}
    write(Path(str(out)+'.validation.json'),result)
    if bad or crc:raise RuntimeError('Collected archive verification failed')
    return result
