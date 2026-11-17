#!/usr/bin/env python3
"""Prepare public atomic datasets from downloaded archives or atomic files."""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from common.runtime import bootstrap
bootstrap()
from data.prepare import prepare

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,default=Path('downloads'))
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('DATA_ROOT','data')))
    p.add_argument('--datasets',nargs='+',choices=['yelp2018','amazon-books','ml-1m'],default=['yelp2018','amazon-books','ml-1m'])
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:
        print(json.dumps({'datasets':a.datasets,'archive_root':str(a.archive_root),'data_root':str(a.data_root),'action':'verify archive SHA-256, convert if required, write preparation manifest'},indent=2));return
    print(json.dumps(prepare(a.data_root,a.archive_root,a.datasets),indent=2))
if __name__=='__main__':main()
