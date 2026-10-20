"""RLlib2.5 Gymnasium environment in isolated Python3.10 training process."""
from pathlib import Path
import json,os,subprocess
import gymnasium as gym
import numpy as np

class BridgeEnv(gym.Env):
 metadata={}
 def __init__(self,config):
  self.config=dict(config);root=Path(config['root'])
  paths=[root/name for name in ['src','scripts','runtime','upstream','training','hpc']]
  env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,paths)),CUDA_VISIBLE_DEVICES='')
  log=Path(config['model_dir'])/'simulation.log';self.log=log.open('a')
  self.process=subprocess.Popen([config['simulation_python'],'-u',str(root/'training/simulation_server.py'),json.dumps(dict(config))],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
  first=self.receive()
  self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(int(first['obs_dim']),),dtype=np.float32)
  self.action_space=gym.spaces.Discrete(int(first['n_actions']))
 def receive(self):
  line=self.process.stdout.readline()
  if not line:raise RuntimeError('Simulator exited; inspect simulation.log')
  result=json.loads(line)
  if 'error' in result:raise RuntimeError(result['error'])
  return result
 def ask(self,msg):
  self.process.stdin.write(json.dumps(msg)+'\n');self.process.stdin.flush();return self.receive()
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed);r=self.ask(dict(op='reset',seed=seed));return np.asarray(r['obs'],np.float32),r['info']
 def step(self,action):
  r=self.ask(dict(op='step',action=int(action)))
  return np.asarray(r['obs'],np.float32),r['reward'],r['terminated'],r['truncated'],r['info']
 def close(self):
  if self.process.poll() is None:
   try:self.process.stdin.write('{"op":"close"}\n');self.process.stdin.flush();self.process.wait(timeout=30)
   except (OSError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
  self.log.close()
