"""Pure exploratory statistics. Bootstrap draws never represent training repeats."""
from __future__ import annotations
import hashlib
import math
import numpy as np
from scipy import stats


def paired_users(users_a, users_b):
    a=np.asarray(users_a,dtype=np.int64);b=np.asarray(users_b,dtype=np.int64)
    if len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b):
        raise ValueError('Multiple slates require explicit user-cluster reduction before this API')
    ia=np.argsort(a,kind='stable');ib=np.argsort(b,kind='stable')
    if not np.array_equal(a[ia],b[ib]):raise ValueError('Evaluation user sets differ. Inner join is prohibited.')
    return ia,ib


def tau_b(x,y):
    """Vectorized exact-tie tau-b, equivalent to scipy.stats.kendalltau variant=b."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.shape!=y.shape or x.ndim<1:raise ValueError('Unmatched method vectors')
    i,j=np.triu_indices(x.shape[-1],1)
    sx=np.sign(x[...,i]-x[...,j]);sy=np.sign(y[...,i]-y[...,j])
    denominator=np.sqrt(np.sum(sx*sx,axis=-1)*np.sum(sy*sy,axis=-1))
    return np.divide(np.sum(sx*sy,axis=-1),denominator,
                     out=np.full(denominator.shape,np.nan),where=denominator>0)


def tie_counts(x):
    x=np.asarray(x);i,j=np.triu_indices(x.shape[-1],1)
    return {'tied_pairs':int(np.sum(x[i]==x[j])), 'distinct_ranks':int(len(np.unique(x)))}


def practical(x,y,delta):
    x=np.asarray(x);y=np.asarray(y);i,j=np.triu_indices(x.shape[-1],1)
    dx=x[...,i]-x[...,j];dy=y[...,i]-y[...,j]
    sx=np.where(dx>delta,1,np.where(dx < -delta,-1,0))
    sy=np.where(dy>delta,1,np.where(dy < -delta,-1,0))
    informative=(sx!=0)&(sy!=0);n=informative.sum(axis=-1);total=len(i)
    same=((sx==sy)&informative).sum(axis=-1);rev=((sx==-sy)&informative).sum(axis=-1)
    def fraction(value,den):
        return np.divide(value,den,out=np.full(np.shape(value),np.nan,dtype=float),where=np.asarray(den)>0)
    return {'total_pairs':np.full(np.shape(n),total),'informative_pairs':n,
            'informative_fraction':fraction(n,total),'sign_agreement_conditional':fraction(same,n),
            'reversal_conditional':fraction(rev,n),'reversal_total':fraction(rev,total),
            'tie_either_fraction':fraction((~informative).sum(axis=-1),total),
            'tie_both_fraction':fraction(((sx==0)&(sy==0)).sum(axis=-1),total)}


def interval(values,level=.95):
    x=np.asarray(values,dtype=float);valid=x[np.isfinite(x)]
    return {'ci_low':float(np.quantile(valid,(1-level)/2)) if len(valid) else None,
            'ci_high':float(np.quantile(valid,1-(1-level)/2)) if len(valid) else None,
            'valid_replicates':len(valid),'undefined_fraction':float(1-len(valid)/len(x)) if len(x) else None}


def bootstrap_macros(matrix,users,replicates=2000,seed=8702026,identity=''):
    """Equal-user macro, paired user clusters, bounded 32-draw batches."""
    matrix=np.asarray(matrix,dtype=np.float64);users=np.asarray(users,dtype=np.int64)
    if matrix.shape[0]!=len(users):raise ValueError('User/metric alignment mismatch')
    unique,inverse=np.unique(users,return_inverse=True)
    # One or multiple slates receive equal weight within each user.
    pooled=np.zeros((len(unique),matrix.shape[1]),dtype=np.float64)
    np.add.at(pooled,inverse,matrix)
    pooled/=np.bincount(inverse)[:,None]
    seed_bytes=hashlib.sha256((str(seed)+'|'+identity).encode()).digest()
    rng=np.random.default_rng(np.random.SeedSequence(np.frombuffer(seed_bytes,dtype=np.uint32)))
    out=np.empty((replicates,matrix.shape[1]),dtype=np.float64)
    n=len(unique)
    for start in range(0,replicates,32):
        count=min(32,replicates-start)
        indices=rng.integers(0,n,size=(count,n))
        counts=np.zeros((count,n),dtype=np.float64)
        for j in range(count):counts[j]=np.bincount(indices[j],minlength=n)
        out[start:start+count]=counts@pooled/n
    return out


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


def paired_tost(effect_target,effect_source,delta):
    """TOST on paired training-seed effects. Zero variance requires structural review."""
    target=np.asarray(effect_target,dtype=float);source=np.asarray(effect_source,dtype=float)
    if target.shape!=source.shape:raise ValueError('Training repeats are not paired')
    z=target-source;out=repeat_interval(z,.95);ci90=repeat_interval(z,.90)
    out.update(delta_raw=delta,delta_x100=100*delta,ci90_low_raw=ci90['ci_low_raw'],
               ci90_high_raw=ci90['ci_high_raw'],ci95_low_raw=out['ci_low_raw'],ci95_high_raw=out['ci_high_raw'])
    if len(z)<2 or np.std(z,ddof=1)==0:
        out.update(tost_status='STRUCTURAL_REVIEW_ZERO_VARIANCE' if len(z)>1 else 'INSUFFICIENT_REPEATS',
                   p_lower=None,p_upper=None,p_equivalence=None,equivalent_unadjusted=None)
        return out
    se=float(z.std(ddof=1)/np.sqrt(len(z)));df=len(z)-1
    lo=float(stats.t.sf((z.mean()+delta)/se,df));hi=float(stats.t.cdf((z.mean()-delta)/se,df))
    out.update(tost_status='COMPUTED',p_lower=lo,p_upper=hi,p_equivalence=max(lo,hi),
               equivalent_unadjusted=bool(max(lo,hi)<.05))
    return out


def holm_planned(p_values,planned=24):
    """Missing prespecified hypotheses carry p=1, never an apparent family completion."""
    if len(p_values)>planned:raise ValueError('More hypotheses than the registered family')
    vals=np.asarray([1. if p is None else float(p) for p in p_values]+[1.]*(planned-len(p_values)))
    order=np.argsort(vals,kind='stable');adjusted=np.empty(planned)
    adjusted[order]=np.minimum(1.,np.maximum.accumulate((planned-np.arange(planned))*vals[order]))
    return adjusted[:len(p_values)]
