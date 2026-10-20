"""Create a transportable result archive and SHA256 manifest; include partial attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from common import ROOT, atomic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--include-models',action='store_true')
    a=p.parse_args()
    if a.archive.exists():raise SystemExit('Refuse to overwrite existing archive')
    files=[f for f in a.out.rglob('*') if f.is_file() and f.suffix in ['.json','.csv','.log','.md']]
    if a.include_models:
        files += [f for f in a.out.rglob('*.pt') if f.is_file()]
        files += [f for f in a.out.glob('models/*/*.npz') if f.is_file()]
        files += [f for f in a.out.glob('models/*/actions.npy') if f.is_file()]
    if not files:raise SystemExit('No result files')
    checksums={str(f.relative_to(a.out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    with tarfile.open(a.archive,'w:gz') as tar:
        for file in sorted(files):tar.add(file,arcname='results/'+str(file.relative_to(a.out)))
        for folder in ['configs','experiments','hpc','docs']:
            for file in sorted((ROOT/folder).rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts:tar.add(file,arcname='provenance/'+str(file.relative_to(ROOT)))
    atomic(a.archive.with_suffix(a.archive.suffix+'.manifest.json'),dict(files=checksums,archive=str(a.archive.name),sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),includes_final_models=a.include_models))
    print(a.archive)
    print(a.archive.with_suffix(a.archive.suffix+'.manifest.json'))

if __name__=='__main__':main()
