from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config_path"] = str(path)
    payload["release_root"] = str(path.parents[1])
    return payload


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_json(path: str | Path, payload) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    clean = {k: v for k, v in config.items() if k not in {"config_path", "release_root"}}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
