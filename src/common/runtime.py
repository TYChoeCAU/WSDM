"""Portable runtime paths and faithful RecBole configuration restoration."""
from __future__ import annotations
import copy
import importlib.util
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
PIN = 'dd3c7f724037b40c04387b031b61fe8715a9e73c'

def bootstrap(require_upstream=False):
    src = REPO / 'src'
    upstream = Path(os.environ.get('NPGC_ROOT', REPO / 'third_party' / 'NPGC')).expanduser().resolve()
    if (upstream / 'recbole').is_dir() and str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    if str(src) in sys.path: sys.path.remove(str(src))
    sys.path.insert(0, str(src))
    if require_upstream and importlib.util.find_spec('recbole') is None:
        raise RuntimeError('RecBole/NPGC is unavailable. Run python scripts/setup_third_party.py or set NPGC_ROOT to the patched pinned checkout.')
    return upstream

def restore_config(snapshot, data_root, checkpoint_dir=None, device='cpu'):
    """Restore saved final values; only runtime paths/device are relocated."""
    bootstrap(True)
    import numpy as np
    for key, value in {'bool': np.bool_, 'int': np.int64, 'float': np.float64,
                       'complex': np.complex128, 'object': np.object_, 'str': np.str_,
                       'long': np.int64, 'unicode': np.str_}.items():
        setattr(np, key, value)
    import torch
    from recbole.config import Config
    from recbole.utils import ModelType, InputType, EvaluatorType
    enum_types = {'ModelType': ModelType, 'InputType': InputType, 'EvaluatorType': EvaluatorType}
    def decode(value):
        if isinstance(value, dict): return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list): return [decode(v) for v in value]
        if isinstance(value, str) and '.' in value:
            kind, member = value.split('.', 1)
            if kind in enum_types and member in enum_types[kind].__members__:
                return enum_types[kind][member]
        return value
    values = decode(copy.deepcopy(snapshot))
    config = Config(model=values['model'], dataset=values['dataset'], config_dict={
        'data_path': str(Path(data_root).resolve()), 'seed': int(values['seed']),
        'use_gpu': str(device).startswith('cuda'), 'gpu_id': str(torch.device(device).index or 0)})
    values.update(data_path=str(Path(data_root).resolve() / values['dataset']),
                  device=torch.device(device), use_gpu=str(device).startswith('cuda'),
                  gpu_id=str(torch.device(device).index or 0), save_dataset=False,
                  save_dataloaders=False, show_progress=False, log_wandb=False,
                  local_rank=0, single_spec=True)
    if checkpoint_dir is not None: values['checkpoint_dir'] = str(Path(checkpoint_dir).resolve())
    config.final_config_dict = {**config.final_config_dict, **values}
    return config
