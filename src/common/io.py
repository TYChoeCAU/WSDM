"""Shared serialization helpers."""
from pathlib import Path
import torch
from diagnostics.revision_evidence.common import atomic_json, read_csv, write_csv

def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

