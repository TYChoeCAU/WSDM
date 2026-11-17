"""Original repeat-interval implementation shared by S1/S2/S3."""
import math
import numpy as np
from scipy import stats

def repeat_interval(values,level=.95):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite actual training repeat')
    n=len(x);mean=float(x.mean()) if n else None
    sd=float(x.std(ddof=1)) if n>1 else None
    half=float(stats.t.ppf((1+level)/2,n-1)*sd/math.sqrt(n)) if n>1 else None
    return dict(n_training_seeds=n,df=n-1 if n else None,mean_raw=mean,sample_sd_raw=sd,
                ci_level=level,ci_low_raw=mean-half if half is not None else None,
                ci_high_raw=mean+half if half is not None else None,
                mean_x100=100*mean if mean is not None else None,
                sample_sd_x100=100*sd if sd is not None else None,
                ci_low_x100=100*(mean-half) if half is not None else None,
                ci_high_x100=100*(mean+half) if half is not None else None)

