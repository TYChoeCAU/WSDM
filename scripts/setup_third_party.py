#!/usr/bin/env python3
"""Fetch the pinned public NPGC code and apply the two recorded local patches."""
import argparse
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/zyouyang/RecSys2025_NonParamGC.git'
PIN = 'dd3c7f724037b40c04387b031b61fe8715a9e73c'
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT/'third_party/NPGC')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); target = args.destination.resolve()
    patches = sorted((ROOT/'patches').glob('*.patch'))
    if args.dry_run:
        print('git clone', URL, target); print('checkout', PIN)
        for path in patches: print('apply', path.name)
        return
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', URL, str(target)], check=True)
    if not (target/'.git').exists(): raise RuntimeError('destination is not a git checkout')
    actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=target, text=True).strip()
    if actual != PIN:
        dirty = subprocess.check_output(['git','status','--porcelain'], cwd=target, text=True)
        if dirty: raise RuntimeError('refusing to change an existing dirty checkout')
        subprocess.run(['git','checkout','--detach',PIN], cwd=target, check=True)
    for patch in patches:
        already = subprocess.run(['git','apply','--reverse','--check',str(patch)], cwd=target, capture_output=True)
        if already.returncode == 0: continue
        subprocess.run(['git','apply','--check',str(patch)], cwd=target, check=True)
        subprocess.run(['git','apply',str(patch)], cwd=target, check=True)
    print('Pinned NPGC source ready:', target)
if __name__ == '__main__': main()
