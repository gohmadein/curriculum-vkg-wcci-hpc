"""Deployment helpers; no Git executable is required on compute nodes."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
CONFIG=ROOT/'experiments/wcci2022_ppo.yaml'
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def fingerprint():
 h=hashlib.sha256()
 for folder in ['src','scripts','training','upstream','hpc','runtime','configs','experiments']:
  for p in sorted((ROOT/folder).rglob('*')):
   if p.is_file() and p.suffix in ['.py','.json','.yaml','.sh','.sbatch','.txt','.npy']:
    h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
 return h.hexdigest()
def verify():
 import importlib.metadata
 reference=json.loads((ROOT/'hpc/simulation_reference_versions.json').read_text())
 for package,expected in reference.items():
  actual=importlib.metadata.version(package).split('+')[0]
  if actual!=expected:raise RuntimeError(f'Simulation dependency differs:{package}={actual},expected{expected}')
 for name,expected in json.loads((ROOT/'hpc/input_manifest.json').read_text()).items():
  if sha(ROOT/name)!=expected:raise RuntimeError('Frozen input changed:'+name)
 dataset=Path(os.environ['VKG_DATA_ROOT'])/'l2rpn_wcci_2022'
 manifest=json.loads((ROOT/'hpc/dataset_manifest.json').read_text())
 names=sorted(p.name for p in (dataset/'chronics').iterdir() if p.is_dir())
 if names!=manifest['chronic_names']:raise RuntimeError('Dataset order differs')
 for name,expected in manifest['files'].items():
  if sha(dataset/name)!=expected:raise RuntimeError('Dataset file differs:'+name)
 from grid2op_vkg.config import load_config
 cfg=load_config(CONFIG)
 split=json.loads((ROOT/'experiments/wcci2022_split.json').read_text())
 assert len(split['train'])==1507 and len(split['test'])==100
 assert not set(split['train'])&(set(split['test'])|set(split['val']))
 return cfg,split
