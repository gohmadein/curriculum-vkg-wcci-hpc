"""Native GeneralTutor rollouts on every train chronic; shared by both model seeds."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
import argparse,copy,fcntl,json,os,time
from pathlib import Path
from common import ROOT,CONFIG,atomic,fingerprint,sha,verify
ENV=None;TUTOR=None;CALLS=0

def init(actions):
 global ENV,TUTOR
 from grid2op_vkg.config import make_env,load_config
 from curriculumagent.tutor.tutors.general_tutor import GeneralTutor
 from grid2op.Observation import BaseObservation
 ENV=make_env(load_config(CONFIG),0)
 parameters=copy.deepcopy(ENV.parameters)
 parameters.MAX_SIMULATE_PER_STEP=parameters.MAX_SIMULATE_PER_EPISODE=-1
 ENV.change_parameters(parameters)
 TUTOR=GeneralTutor(ENV.action_space,Path(actions),return_status=False)
 original=BaseObservation.simulate
 def counted(self,*args,**kwargs):
  global CALLS
  CALLS+=1;return original(self,*args,**kwargs)
 BaseObservation.simulate=counted

def rollout(job):
 global CALLS
 import numpy as np
 from curriculumagent.common.utilities import split_and_execute_action,find_best_line_to_reconnect
 cid,name,destination,source=job
 ENV.seed(cid);ENV.set_id(name);obs=ENV.reset();TUTOR.next_actions=None
 done=False;data=[];CALLS=0;started=time.monotonic()
 while not done:
  vector,index=TUTOR.act_with_id(obs)
  action=ENV.action_space.from_vect(vector)
  if action.as_dict() and index!=-1:
   features=obs.to_vect().astype(np.float32)
   if not np.isfinite(features).all():raise RuntimeError('Nonfinite Tutor state')
   data.append(np.concatenate((np.array([index],np.float32),features)))
   obs,_,done,_=split_and_execute_action(ENV,action.to_vect())
  else:obs,_,done,_=ENV.step(find_best_line_to_reconnect(obs,ENV.action_space()))
 target=Path(destination)/f'train_{cid}.npz'
 values=np.asarray(data,np.float32).reshape(-1,1+int(ENV.observation_space.n))
 np.savez_compressed(target,rows=values)
 record=dict(status='complete',chronic_id=cid,chronic=name,states=len(values),obs_dim=int(ENV.observation_space.n),steps=int(ENV.nb_time_step),offline_simulates=CALLS,elapsed_s=time.monotonic()-started,source_digest=source,data_sha256=sha(target))
 atomic(target.with_suffix('.json'),record)
 return record

def main():
 import numpy as np
 from grid2op_vkg.config import make_env
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
 if not 1<=a.workers<=18:p.error('Use1..18workers')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg,split=verify();digest=fingerprint()
 marker=out/'manifest.json'
 if marker.exists():
  m=json.loads(marker.read_text())
  if m['source_digest']!=digest:raise RuntimeError('Preparation source changed')
  for r in m['chronics']:assert sha(out/'parts'/f"train_{r['chronic_id']}.npz")==r['data_sha256']
  print('Preparation already complete');return
 parts=out/'parts';parts.mkdir(exist_ok=True)
 env=make_env(cfg,0)
 try:
  names=sorted(Path(str(x)).name for x in env.chronics_handler.real_data.subpaths)
  library=np.load(ROOT/'experiments/wcci2022_library.npy')
  actions=np.array([env.action_space({'set_bus':v.astype(int)}).to_vect() for v in library])
  assert actions.shape[0]==387
  np.save(out/'actions.npy',actions)
 finally:env.close()
 jobs=[];completed=[]
 for cid in split['train']:
  f=parts/f'train_{cid}.json';data=f.with_suffix('.npz')
  if f.exists() and data.exists():
   record=json.loads(f.read_text())
   if record['source_digest']==digest and sha(data)==record['data_sha256']:completed.append(record);continue
  jobs.append((cid,names[cid],str(parts),digest))
 status=dict(status='running',source_digest=digest,config_sha=cfg['_sha256'],total=1507,complete=len(completed),started=datetime.now().isoformat())
 atomic(out/'status.json',status)
 try:
  with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(out/'actions.npy'),)) as pool:
   for future in as_completed([pool.submit(rollout,job) for job in jobs]):
    completed.append(future.result());status['complete']=len(completed);atomic(out/'status.json',status)
    print('Tutor',len(completed),'/1507',flush=True)
  assert {r['chronic_id'] for r in completed}==set(split['train'])
  assert sum(r['states'] for r in completed)>0
  meta=dict(status='complete',source_digest=digest,config_sha=cfg['_sha256'],train_ids=split['train'],train_chronics_sha256=__import__('hashlib').sha256(json.dumps(sorted(split['train'])).encode()).hexdigest(),actions_sha256=sha(out/'actions.npy'),obs_dim=completed[0]['obs_dim'],action_count=387,states=sum(r['states'] for r in completed),offline_simulates=sum(r['offline_simulates'] for r in completed),chronics=sorted(completed,key=lambda r:r['chronic_id']),protocol=json.loads((ROOT/'hpc/protocol.json').read_text()))
  atomic(marker,meta);status.update(status='complete',finished=datetime.now().isoformat())
 except Exception as exc:status.update(status='failed',reason=str(exc));raise
 finally:atomic(out/'status.json',status)

if __name__=='__main__':main()
