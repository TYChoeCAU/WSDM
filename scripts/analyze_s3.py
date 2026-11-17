#!/usr/bin/env python3
"""Recompute S3 statistics from generated saved predictions, without inference."""
import sys
from run_s3 import main
if __name__=='__main__':
    sys.argv[1:1]=['--stage','statistics']
    main()
