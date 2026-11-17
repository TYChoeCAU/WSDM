"""Public S3 paths, fixed scope, atomic I/O, and configuration integrity."""
from pathlib import Path
import hashlib,json,csv,os
ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/'configs/s3'
RUN=Path(os.environ.get('S3_RUN',ROOT/'artifacts/s3')).expanduser().resolve()
DATA_ROOT=Path(os.environ.get('DATA_ROOT',ROOT/'dataset')).expanduser().resolve()
MATCHED=RUN
DATASETS=('yelp2018','ml-1m')
BACKBONES=('NFM','MF_BPR')
SEEDS=(2020,2021,2022,2023,2024)
PROTOCOLS=('U','W','F')
FIXED_DATA_SEED=2020
MASKS={'S':(1,0,0,0,0,0,0),'P':(1,1,0,0,0,0,0),'R':(1,0,1,1,0,0,0),'PR':(1,1,1,1,0,0,0),
       'H':(1,0,0,0,1,1,1),'RH':(1,0,1,1,1,1,1),'PH':(1,1,0,0,1,1,1),'PRH':(1,1,1,1,1,1,1)}
FEATURES=('s','log1p_item_degree','log1p_q','n_eff','h','h_minus_s','variance')
METHODS=('base','item_degree_ranking','original_nonparamgc','prm_principled',*MASKS)
LEARNED=('prm_principled',*MASKS)
class UnlimitedBudget:
    def isoformat(self):return None
DEADLINE=UnlimitedBudget()
def deadline(*args,**kwargs):return None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def read_csv(path):
    with Path(path).open(newline='')as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    rows=list(rows);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))or['status'];q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with q.open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    q.replace(p)
def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
def save(path,value):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    torch.save(value,q);q.replace(p)
def coord(dataset,backbone,seed,protocol=None,method=None,area='runs'):
    p=RUN/area/dataset/backbone/f'seed_{int(seed)}'
    if protocol is not None:p/=protocol
    if method is not None:p/=method
    return p
def cache_dir(dataset,backbone,seed,protocol):return coord(dataset,backbone,seed,protocol,area='cache')
def verify_configuration():
    manifest=json.loads((CONFIG/'manifest.json').read_text())
    for name,checksum in manifest['files'].items():
        if sha(CONFIG/name)!=checksum:raise ValueError('Immutable S3 configuration changed: '+name)
    return manifest
