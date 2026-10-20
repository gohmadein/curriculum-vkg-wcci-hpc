"""Native Junior architecture/NLL pretraining then native Torch Senior via RLlib2.5.1."""
import argparse,collections,hashlib,json,os,random,signal,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from native_model import Grid2OpCustomModelTorch
from bridge import BridgeEnv

STOP=False
def stopped(signum,frame):
 global STOP
 STOP=True

def atomic(path,data):
 p=Path(path);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def junior_network(dim,actions):
 # Matches upstream Junior_PtL default four1000-unit ReLU layers and two0.25dropouts.
 layers=[]
 for i in range(1,5):
  layers.extend([(f'layer{i}',nn.Linear(dim if i==1 else 1000,1000)),(f'activation{i}',nn.ReLU())])
  if i in [3,4]:layers.append((f'dropout{i-2}',nn.Dropout(.25)))
 layers.append(('act_layer',nn.Linear(1000,actions)))
 net=nn.Sequential(collections.OrderedDict(layers))
 for module in net.modules():
  if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight)
 return net

def load_training(prep):
 manifest=json.loads((prep/'manifest.json').read_text())
 rows=[]
 for record in manifest['chronics']:
  path=prep/'parts'/f"train_{record['chronic_id']}.npz"
  if digest(path)!=record['data_sha256']:raise RuntimeError('Tutor shard checksum differs')
  a=np.load(path,allow_pickle=False)['rows']
  if len(a):rows.append(a)
 values=np.concatenate(rows)
 x=values[:,1:].astype(np.float32);y=values[:,0].astype(np.int64)
 assert x.shape[1]==manifest['obs_dim'] and np.isfinite(x).all()
 assert y.min()>=0 and y.max()<manifest['action_count']
 return manifest,x,y

def main():
 global STOP
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--simulation-python',required=True);p.add_argument('--seed',type=int,choices=[0,1],required=True);p.add_argument('--source-digest',required=True)
 a=p.parse_args();root=a.root.resolve();prep=a.preparation.resolve();out=a.model_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 for sig in [signal.SIGTERM,signal.SIGUSR1,signal.SIGINT]:signal.signal(sig,stopped)
 if np.__version__!='1.23.5' or torch.__version__.split('+')[0]!='2.8.0' or __import__('ray').__version__!='2.5.1':raise RuntimeError('Training dependency versions differ from pinned protocol')
 if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one Slurm-assignedGPU')
 torch.set_num_threads(1);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 plan=json.loads((root/'hpc/protocol.json').read_text())
 prepmeta,x,y=load_training(prep)
 statefile=out/'run_meta.json'
 state=dict(status='running',seed=a.seed,source_digest=a.source_digest,config_sha=prepmeta['config_sha'],train_chronics_sha256=prepmeta['train_chronics_sha256'],actions_sha256=prepmeta['actions_sha256'],target_senior_iterations=plan['senior']['iterations'],observation_dim=int(x.shape[1]),action_count=int(prepmeta['action_count']),junior_epochs=0,senior_iterations=0,started=datetime.now().isoformat(),learner='CurriculumAgent-adapted/native Torch Junior + RLlib2.5.1 Senior',gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,ray_version=__import__('ray').__version__,protocol=plan)
 if statefile.exists():
  previous=json.loads(statefile.read_text())
  if previous['source_digest']!=a.source_digest:raise RuntimeError('Refuse continuation after source change')
  if previous['status']=='complete':print('Already complete');return
  state.update(previous,status='running')
 atomic(statefile,state)
 started=time.monotonic()
 try:
  checkpoint=out/'junior_checkpoint.pt';junior=junior_network(x.shape[1],prepmeta['action_count']).cuda();optimizer=torch.optim.Adam(junior.parameters(),lr=plan['junior']['learning_rate'])
  epoch0=0
  if checkpoint.exists():
   saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
   junior.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);epoch0=saved['epoch']
   torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
  loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=plan['junior']['batch_size'],shuffle=False,num_workers=0)
  for epoch in range(epoch0,plan['junior']['epochs']):
   junior.train();total=0.;count=0
   for features,labels in loader:
    optimizer.zero_grad();scores=junior(features.cuda());loss=nn.functional.nll_loss(nn.functional.log_softmax(scores,dim=1),labels.cuda());loss.backward();optimizer.step();total+=float(loss.detach())*len(labels);count+=len(labels)
   torch.save(dict(model=junior.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'junior_checkpoint.tmp')
   (out/'junior_checkpoint.tmp').replace(checkpoint)
   state.update(stage='junior',junior_epochs=epoch+1,junior_loss=total/count,elapsed_s=time.monotonic()-started);atomic(statefile,state)
   print('Junior epoch',epoch+1,'loss',total/count,flush=True)
   if STOP:raise InterruptedError('Scheduler signal; Junior epoch checkpoint saved')
  weights={k:v.detach().cpu() for k,v in junior.state_dict().items()}
  torch.save(weights,out/'junior_final.pt')
  parity_sample=x[:min(32,len(x))].copy()
  del optimizer,junior,loader,x,y;torch.cuda.empty_cache()
  import ray
  from ray.rllib.algorithms.ppo import PPOConfig
  from ray.rllib.models import ModelCatalog
  from ray.tune.registry import register_env
  from ray.tune.logger import UnifiedLogger
  import tempfile
  temp=Path(tempfile.mkdtemp(prefix=f'cvkg_s{a.seed}_',dir='/tmp'))
  state['ray_temp_dir']=str(temp);atomic(statefile,state)
  ray.init(num_cpus=1,num_gpus=1,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=str(temp),object_store_memory=1024**3,ignore_reinit_error=False)
  ModelCatalog.register_custom_model('CurriculumNativeTorch',Grid2OpCustomModelTorch)
  register_env('FrozenCurriculumBridge',lambda config:BridgeEnv(config))
  config=dict(root=str(root),config=str(root/'experiments/wcci2022_ppo.yaml'),split_file=str(root/'experiments/wcci2022_split.json'),actions=str(prep/'actions.npy'),seed=a.seed,model_dir=str(out),simulation_python=a.simulation_python)
  cfg=(PPOConfig().environment(env='FrozenCurriculumBridge',env_config=config)
       .framework('torch').rollouts(num_rollout_workers=0,create_env_on_local_worker=True,rollout_fragment_length=200)
       .resources(num_gpus=1).debugging(seed=a.seed)
       .training(model={'custom_model':'CurriculumNativeTorch','custom_model_config':{'model_path':str(out/'junior_final.pt')}},train_batch_size=4000,sgd_minibatch_size=128,num_sgd_iter=30,lr=.00005,gamma=.99))
  def logger_creator(configuration):
   folder=out/'ray_logs';folder.mkdir(exist_ok=True);return UnifiedLogger(configuration,str(folder),loggers=None)
  algorithm=cfg.build(logger_creator=logger_creator)
  try:
   cpfile=out/'senior_checkpoint.json'
   if cpfile.exists():
    checkpoint_info=json.loads(cpfile.read_text());algorithm.restore(checkpoint_info['path']);iteration0=int(checkpoint_info['iteration'])
   else:iteration0=0
   for iteration in range(iteration0,plan['senior']['iterations']):
    result=algorithm.train()
    if not np.isfinite(result.get('episode_reward_mean',0.)):raise RuntimeError('Nonfinite Senior return')
    if (iteration+1)%5==0 or iteration+1==plan['senior']['iterations'] or STOP:
     destination=out/'checkpoints';destination.mkdir(exist_ok=True)
     saved=algorithm.save(str(destination))
     path=str(saved) if isinstance(saved,(str,Path)) else str(saved.checkpoint.path)
     atomic(cpfile,dict(path=path,iteration=iteration+1))
    state.update(stage='senior',senior_iterations=iteration+1,senior_timesteps=int(result.get('timesteps_total',0)),episode_reward_mean=float(result.get('episode_reward_mean',0)),elapsed_s=time.monotonic()-started)
    atomic(statefile,state);print('Senior iteration',iteration+1,'timesteps',state['senior_timesteps'],flush=True)
    if STOP:raise InterruptedError('Scheduler signal; standard RLlib checkpoint saved')
   policy=algorithm.get_policy();model=policy.model;export={}
   for i in range(4):
    layer=getattr(model,f'layer{i+1}');export[f'W{i}']=layer.weight.detach().cpu().numpy().T;export[f'b{i}']=layer.bias.detach().cpu().numpy()
   export['W_act']=model.act_layer.weight.detach().cpu().numpy().T;export['b_act']=model.act_layer.bias.detach().cpu().numpy()
   # Export parity is measured on training states only, never test selection.
   sample=parity_sample;h=sample.copy()
   for i in range(4):h=np.maximum(h@export[f'W{i}']+export[f'b{i}'],0)
   logits=h@export['W_act']+export['b_act'];exp=np.exp(logits-logits.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
   model.eval()
   with torch.no_grad():native,_=model({'obs':torch.as_tensor(sample,device=next(model.parameters()).device)},[],None)
   error=float(np.max(np.abs(prob-native.detach().cpu().numpy())))
   if error>1e-4:raise RuntimeError(f'Export parity failed:{error}')
   np.savez_compressed(out/'policy.npz',**export)
   import shutil
   shutil.copyfile(prep/'actions.npy',out/'actions.npy')
   simulator=algorithm.workers.local_worker().env
   stats=simulator.ask(dict(op='stats'))
   if not set(stats['visited_train_chronics'])<=set(prepmeta['train_ids']):raise RuntimeError('Senior visited non-train chronic')
   state.update(status='complete',finished=datetime.now().isoformat(),policy_sha256=digest(out/'policy.npz'),export_max_probability_error=error,training_stats=stats,elapsed_s=time.monotonic()-started)
   atomic(statefile,state)
  finally:algorithm.stop();ray.shutdown()
 except Exception as exc:
  state.update(status='interrupted' if isinstance(exc,InterruptedError) else 'failed',reason=str(exc));atomic(statefile,state);raise

if __name__=='__main__':main()
