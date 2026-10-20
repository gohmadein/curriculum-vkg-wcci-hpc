"""One shared Senior per seed, then all four100-chronic evaluation cells."""
from pathlib import Path
from datetime import datetime
import argparse,csv,fcntl,json,math,os,signal,subprocess,sys
from common import ROOT,CONFIG,atomic,fingerprint,sha,verify
CHILD=None

def interrupt(signum,frame):
 if CHILD is not None and CHILD.poll() is None:
  CHILD.send_signal(signal.SIGUSR1)
  try:CHILD.wait(timeout=90)
  except subprocess.TimeoutExpired:CHILD.terminate();CHILD.wait(timeout=20)
 raise InterruptedError('Scheduler signal; outputs/checkpoints retained')

def main():
 global CHILD
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[0,1],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--training-python',required=True);p.add_argument('--eval-workers',type=int,default=18);a=p.parse_args()
 if not 1<=a.eval_workers<=18:p.error('Use1..18evaluation workers')
 out=a.out.resolve();model=out/f'models/senior_s{a.seed}';model.mkdir(parents=True,exist_ok=True)
 lock=(model/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,interrupt)
 cfg,split=verify();source=fingerprint();prep=out/'preparation';prepmeta=json.loads((prep/'manifest.json').read_text())
 if prepmeta['source_digest']!=source:raise RuntimeError('Preparation/code fingerprint differs')
 state=dict(seed=a.seed,status='running',started=datetime.now().isoformat(),source_digest=source,config_sha=cfg['_sha256'])
 statefile=out/f'status_s{a.seed}.json'
 try:
  metadata=model/'run_meta.json'
  finished=json.loads(metadata.read_text()) if metadata.exists() else {}
  if finished.get('status')!='complete':
   cmd=[a.training_python,'-u',str(ROOT/'training/train.py'),'--root',str(ROOT),'--preparation',str(prep),'--model-dir',str(model),'--simulation-python',sys.executable,'--seed',str(a.seed),'--source-digest',source]
   state.update(stage='train',command=cmd);atomic(statefile,state)
   env=dict(os.environ,PYTHONPATH=str(ROOT/'training'))
   with (model/'training.log').open('a') as log:
    CHILD=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=CHILD.wait()
   if code:raise RuntimeError(f'Training exited{code}; see{model}/training.log')
  meta=json.loads(metadata.read_text())
  assert meta['status']=='complete' and meta['source_digest']==source and meta['config_sha']==cfg['_sha256']
  assert meta['seed']==a.seed and meta['junior_epochs']==100 and meta['senior_iterations']==50
  assert sha(model/'policy.npz')==meta['policy_sha256'] and sha(model/'actions.npy')==prepmeta['actions_sha256']
  from run_factorial import read_rows,write_csv
  for cell in 'ABCD':
   folder=out/f'eval/{cell}_s{a.seed}';folder.mkdir(parents=True,exist_ok=True)
   marker=folder/'completed.json'
   if marker.exists():
    assert json.loads(marker.read_text())['source_digest']==source
    continue
   attempt=folder/datetime.now().strftime('%Y%m%dT%H%M%S_%f');attempt.mkdir()
   label=f'Curriculum-WCCI:{cell}_s{a.seed}'
   kwargs={label:dict(model_dir=str(model),cell=cell)}
   cmd=[sys.executable,'scripts/eval_baselines.py','--config',str(CONFIG),'--agent-module','grid2op_vkg.vkg.curriculum','--agents',label,'--agent-kwargs',json.dumps(kwargs),'--split','test','--chronics',*map(str,split['test']),'--seed','0','--workers',str(a.eval_workers),'--out',str(attempt)]
   state.update(stage='evaluate',cell=cell,attempt=str(attempt));atomic(statefile,state)
   audit=attempt/'audit'
   with (attempt/'harness.log').open('w') as log:
    CHILD=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',VKG_CURRICULUM_AUDIT_DIR=str(audit)),stdout=log,stderr=subprocess.STDOUT);code=CHILD.wait()
   if code:raise RuntimeError(f'Evaluation exited{code}; see{attempt}/harness.log')
   rows=read_rows(attempt)
   assert len(rows)==100 and {int(r['chronic_id']) for r in rows}==set(split['test'])
   for r in rows:
    assert not r['agent_error'] and int(r['n_agent_errors'])==0 and int(r['horizon'])==2017
    assert math.isfinite(float(r['survival'])) and abs(float(r['survival'])-int(r['steps'])/2017)<1e-9
    record=json.loads((audit/f"{cell}_s{a.seed}__{r['chronic']}.json").read_text())
    assert record['semantic_supplement']==(cell in 'BD') and record['rewrite_dead']==(cell in 'CD')
    assert record['max_simulate_per_step']<=40 and record['simulate_calls']==int(r['n_simulate'])
   write_csv(folder/'per_episode_verified.csv',[dict(r,cell=cell,training_seed=a.seed) for r in rows])
   if fingerprint()!=source:raise RuntimeError('Code changed while job running')
   atomic(marker,dict(status='complete',cell=cell,seed=a.seed,source_digest=source,model=str(model),attempt=str(attempt),episodes=100))
  state.update(status='complete',finished=datetime.now().isoformat())
 except Exception as exc:state.update(status='interrupted' if isinstance(exc,InterruptedError) else 'failed',reason=str(exc));raise
 finally:atomic(statefile,state)

if __name__=='__main__':main()
