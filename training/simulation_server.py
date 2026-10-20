"""Local stdio JSON bridge; no network port or authentication material."""
import contextlib,json,sys,traceback
from simulation_env import SimulationEnv
protocol=sys.stdout
with contextlib.redirect_stdout(sys.stderr):
 env=SimulationEnv(json.loads(sys.argv[1]))
protocol.write(json.dumps(dict(ready=True,obs_dim=env.obs_dim,n_actions=len(env.actions)))+'\n');protocol.flush()
try:
 for line in sys.stdin:
  msg=json.loads(line)
  try:
   with contextlib.redirect_stdout(sys.stderr):
    if msg['op']=='reset':
     obs,info=env.reset(msg.get('seed'));reply=dict(obs=obs.tolist(),info=info)
    elif msg['op']=='step':
     obs,reward,term,trunc,info=env.step(msg['action']);reply=dict(obs=obs.tolist(),reward=reward,terminated=term,truncated=trunc,info=info)
    elif msg['op']=='stats':reply=env.stats()
    elif msg['op']=='close':break
    else:raise ValueError(msg['op'])
   protocol.write(json.dumps(reply,allow_nan=False)+'\n');protocol.flush()
  except Exception:
   protocol.write(json.dumps(dict(error=traceback.format_exc()))+'\n');protocol.flush();raise
finally:env.close()
