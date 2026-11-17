"""Explicit artifact paths. No historic server path is inferred or executed."""
from pathlib import Path
import json, os
from common.runtime import REPO
ROOT=REPO
OUT=Path(os.environ.get('FOLLOWUP_OUT',REPO/'artifacts/followup')).expanduser().resolve()
S3=Path(os.environ.get('S3_RUN',REPO/'artifacts/s3')).expanduser().resolve()
S2=Path(os.environ.get('S2_RUN',REPO/'artifacts/s2')).expanduser().resolve()
DATA_ROOT=Path(os.environ.get('DATA_ROOT',REPO/'dataset')).expanduser().resolve()
EVIDENCE=Path(os.environ.get('FOLLOWUP_EVIDENCE',REPO/'results')).expanduser().resolve()
MANIFEST=Path(os.environ['FOLLOWUP_MANIFEST']).expanduser().resolve() if os.environ.get('FOLLOWUP_MANIFEST') else None

def manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST is not None else {}

def resolve(value):
    """Relative manifest paths resolve against its directory, never the old project."""
    if value is None: raise ValueError('A required artifact path is missing')
    value=manifest().get('path_map',{}).get(str(value),value)
    if str(value).startswith(('artifact:','historical/')):
        raise ValueError('Logical provenance IDs are not input files. Supply a real local path in the manifest: '+str(value))
    path=Path(value).expanduser()
    return (path if path.is_absolute() else (MANIFEST.parent if MANIFEST else REPO)/path).resolve()

def coordinate(section,dataset,backbone,seed,required=True,**extra):
    rows=manifest().get(section,[])
    selected=[r for r in rows if (r['dataset'],r['backbone'],int(r['seed']))==(dataset,backbone,int(seed)) and all(r.get(k)==v for k,v in extra.items())]
    if len(selected)>1 or (required and len(selected)!=1):raise ValueError('Expected one manifest '+section+' coordinate '+str((dataset,backbone,seed,extra)))
    return selected[0] if selected else None

def input_file(name,default=None):
    value=manifest().get('inputs',{}).get(name)
    return resolve(value) if value is not None else Path(default) if default is not None else resolve(None)

def s1_spec(dataset,backbone,seed):
    return json.loads((REPO/'configs/s1'/dataset/backbone/f'seed_{int(seed)}.json').read_text())
