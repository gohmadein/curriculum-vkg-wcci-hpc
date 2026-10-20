"""Frozen Grid2Op simulator plus native Senior rollout methods, Python3.13 side."""
import logging
from pathlib import Path
import numpy as np
from native_env_methods import NativeSeniorMethods
from curriculumagent.senior.rllib_execution.alternative_rewards import PPO_Reward
from grid2op_vkg.config import load_config,make_env,_reward_cls

class RewardProxy:
 def __init__(self,env):self.env=env;self.env_steps=0;self.illegal=0;self.ambiguous=0
 def __getattr__(self,name):return getattr(self.env,name)
 def step(self,action):
  obs,reward,done,info=self.env.step(action)
  self.env_steps+=1;self.illegal+=int(info.get('is_illegal',False));self.ambiguous+=int(info.get('is_ambiguous',False))
  return obs,float(info['rewards']['curriculum']),done,info

class SimulationEnv(NativeSeniorMethods):
 def __init__(self,config):
  import json
  cfg=load_config(config['config'])
  others={k:_reward_cls(v) for k,v in cfg['env'].get('other_rewards',{}).items()};others['curriculum']=PPO_Reward
  env=make_env(cfg,int(config['seed']),other_rewards=others)
  self.single_env=RewardProxy(env)
  self.names=sorted(Path(str(x)).name for x in env.chronics_handler.real_data.subpaths)
  self.train_ids=json.loads(Path(config['split_file']).read_text())['train']
  self.rng=np.random.default_rng(int(config['seed']));self.seed=int(config['seed'])
  self.actions=np.load(config['actions']);self.obs_dim=int(env.observation_space.n)
  self.subset=False;self.scaler=None;self.topo=False;self.topology_actions=False;self.action_threshold=.95
  self.obs_grid2op=env.get_obs();self.step_in_env=0;self.passiv_reward=0;self.info={};self.done=True
  self.visited=set();self.safe_reset_skips=0;self.last_finite=None
 def reset(self,seed=None,options=None):
  if seed is not None:self.rng=np.random.default_rng(int(seed));self.single_env.seed(int(seed))
  for _ in range(len(self.train_ids)*2):
   cid=int(self.rng.choice(self.train_ids));self.visited.add(cid)
   self.single_env.set_id(self.names[cid]);self.obs_grid2op=self.single_env.reset()
   self.step_in_env=0;self.passiv_reward=0;self.obs_rl=self.run_next_action()
   self.done=bool(self.single_env.done)
   if not self.done:
    if not np.isfinite(self.obs_rl).all():raise RuntimeError('Nonfinite Senior reset')
    self.last_finite=np.asarray(self.obs_rl,np.float32).copy();self.truncated,self.terminated=self.check_terminated_truncated()
    return self.last_finite,{}
   self.safe_reset_skips+=1
  raise RuntimeError('No train chronic requiring Senior intervention after bounded reset attempts')
 def step(self,action):
  obs,reward,terminated,truncated,info=super().step(int(action))
  obs=np.asarray(obs,np.float32)
  if not np.isfinite(obs).all():
   if not (terminated or truncated):raise RuntimeError('Nonfinite nonterminal Senior observation')
   obs=self.last_finite.copy()
  self.last_finite=obs.copy()
  return obs,float(reward),bool(terminated),bool(truncated),{}
 def stats(self):
  return dict(env_steps=self.single_env.env_steps,illegal=self.single_env.illegal,ambiguous=self.single_env.ambiguous,visited_train_chronics=sorted(self.visited),safe_reset_skips=self.safe_reset_skips)
 def close(self):self.single_env.close()
