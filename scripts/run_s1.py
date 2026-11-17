#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common.runtime import bootstrap
bootstrap()
from studies.s1.run import main
if __name__=='__main__':main()
