"""Audit2shared models/8evaluations/800episodes and produce A/B/D/C cells."""
from pathlib import Path
import argparse,csv,json,math,statistics,random
from common import atomic

def load(path):return list(csv.DictReader(path.open()))
def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 rows=[];fingerprints=set();per_seed=[]
 for seed in [0,1]:
  meta=json.loads((out/f'models/senior_s{seed}/run_meta.json').read_text())
  assert meta['status']=='complete' and meta['senior_iterations']==50 and meta['junior_epochs']==100
  for cell in 'ABCD':
   folder=out/f'eval/{cell}_s{seed}';marker=json.loads((folder/'completed.json').read_text());fingerprints.add(marker['source_digest'])
   assert marker['status']=='complete' and marker['seed']==seed and marker['cell']==cell
   rs=load(folder/'per_episode_verified.csv');assert len(rs)==100 and len({r['chronic_id'] for r in rs})==100
   assert all(int(r['n_agent_errors'])==0 and not r['agent_error'] and int(r['horizon'])==2017 for r in rs)
   per_seed.append(dict(cell=cell,seed=seed,survival_pct=100*statistics.mean(float(r['survival']) for r in rs),illegal_per_episode=statistics.mean(int(r['n_illegal']) for r in rs)))
   rows.extend(rs)
 assert len(fingerprints)==1 and len(rows)==800
 cells=[];cohort=None;maps={}
 for cell in 'ABDC':
  rs=[r for r in rows if r['cell']==cell];ids={int(r['chronic_id']) for r in rs}
  if cohort is None:cohort=ids
  assert ids==cohort
  vals=[r['survival_pct'] for r in per_seed if r['cell']==cell]
  cells.append(dict(cell=cell,survival_pct=statistics.mean(vals),seed_sd_pp=statistics.stdev(vals),episodes=200,seeds=2,full=sum(int(r['steps'])==2017 for r in rs),illegal_per_episode=statistics.mean(int(r['n_illegal']) for r in rs),simulate_per_episode=statistics.mean(int(r['n_simulate']) for r in rs)))
  maps[cell]={cid:statistics.mean(float(r['survival']) for r in rs if int(r['chronic_id'])==cid) for cid in ids}
 contrasts=[]
 for x,y in [('B','A'),('C','A'),('D','B'),('D','C'),('D','A')]:
  diff=[100*(maps[x][i]-maps[y][i]) for i in sorted(cohort)];wins=sum(v>1e-9 for v in diff);losses=sum(v< -1e-9 for v in diff);n=wins+losses
  rng=random.Random(20261020);boot=sorted(statistics.mean(rng.choices(diff,k=len(diff))) for _ in range(10000))
  contrasts.append(dict(contrast=f'{x}-{y}',gain_pp=statistics.mean(diff),wins=wins,losses=losses,ties=100-wins-losses,sign_p=1 if n==0 else min(1,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n),scenario_ci95_low=boot[249],scenario_ci95_high=boot[9749]))
 dest=out/'summary';dest.mkdir(exist_ok=True)
 for name,items in [('cells.csv',cells),('per_seed.csv',per_seed),('per_episode_verified.csv',rows),('contrasts.csv',contrasts)]:write(dest/name,items)
 atomic(dest/'summary.json',dict(status='complete',models=2,seeds=[0,1],evaluations=8,episodes=800,cells=cells,contrasts=contrasts,source_digest=next(iter(fingerprints))))
 (dest/'table.md').write_text('|Method|Baseline A|+VKG B|+VKG+TR D|+TR C|\n|---|---:|---:|---:|---:|\n|CurriculumAgent adapted,118-bus|'+'|'.join(f"{r['survival_pct']:.2f} ± {r['seed_sd_pp']:.2f}%" for r in cells)+'|\n\nTwo shared trained seniors; each evaluated in all four cells on100test. Draw seed matches training seed.\n')
 print((dest/'table.md').read_text())
if __name__=='__main__':main()
