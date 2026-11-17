"""Bind completed follow-up code to local public S3 artifacts and implementations."""
from types import SimpleNamespace
from pathlib import Path
import json,time
import numpy as np
from .paths import S3,coordinate,resolve,input_file
from studies.s3 import s3_common as common
from studies.s3.cross_common import sha,identity,load
PARENT=S3

def cache_paths(dataset,backbone,seed,protocol):
    row=coordinate('s3_caches',dataset,backbone,seed,False,protocol=protocol)
    if row is None:return common.cache_paths(dataset,backbone,seed,protocol)
    return {key:resolve(row[key]) for key in ('directory','feature_cache_path','slate_cache_path','complete_path')}

def result_path(dataset,backbone,seed,protocol,method,family='matched'):
    row=coordinate('s3_results',dataset,backbone,seed,False,protocol=protocol,method=method,family=family)
    if row:return resolve(row['result_path'])
    if family!='matched':raise ValueError('Frozen historical result requires an explicit s3_results manifest entry')
    return common.result_path(dataset,backbone,seed,protocol,method)

def backbone_result(dataset,backbone,seed):
    row=coordinate('s3_backbones',dataset,backbone,seed,False)
    return resolve(row['result_path']) if row else S3/'backbones'/dataset/backbone/f'seed_{int(seed)}'/'result.json'

def _neighbors(dataset,graph,fingerprint):
    # Follow-up inference must read existing neighbor tables, never build them.
    from studies.corrected.native import tables
    result={};meta={};start=time.monotonic()
    for family in ('minimal','nonparamgc'):
        path=input_file(dataset+'_'+family+'_neighbors',S3/'master'/dataset/'neighbors'/f'{family}_k2.npz')
        result[family],meta[family]=tables(path,fingerprint,2,family)
    return result,meta,time.monotonic()-start

def install():
    modules=common.configure_runtime()
    modules['features']._neighbors=_neighbors
    return modules,{}

def sources():
    return {f'artifact:source/followup/{p.name}':sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
