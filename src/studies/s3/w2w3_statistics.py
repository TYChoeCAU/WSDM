"""Registered W2/W3 statistics on saved predictions, without fitting or inference."""
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np

from .cross_common import RUN, ROOT
from . import p0_statistics as previous
tau_b = previous.tau_b
bootstrap_macros = previous.bootstrap_macros
interval = previous.interval
repeat_interval = previous.repeat_interval
paired_tost = previous.paired_tost
holm_planned = previous.holm_planned

FIXED = ('base', 'item_degree_ranking', 'original_nonparamgc')
RESIDUALS = ('S', 'P', 'R', 'PR', 'H', 'RH', 'PH', 'PRH')
METHODS = (*FIXED, 'prm_principled', *RESIDUALS)
METHOD_SETS = {'FIXED3': FIXED, 'REF4': METHODS[:4], 'ALL12': METHODS}
CONTRASTS = {
    'Pop-Base': ('item_degree_ranking','base'),
    'NPGC-Base': ('original_nonparamgc','base'),
    'PRM+PL-Base': ('prm_principled','base'),
    'Pop-NPGC': ('item_degree_ranking','original_nonparamgc'),
    'PRM+PL-NPGC': ('prm_principled','original_nonparamgc'),
    'PRM+PL-Res(p)': ('prm_principled','P'),
    'Res(p)-Pop': ('P','item_degree_ranking'),
    'Res(H)-NPGC': ('H','original_nonparamgc'),
}
METRICS = ('ndcg@10','ndcg@20','recall@10','recall@20')
DELTAS = (.001,.0025,.005)

def finite(value):
    value=float(value)
    return value if np.isfinite(value) else None

def classify(source95, target95, source90, target90, delta):
    """Prespecified interval decision, distinct from effect-change Holm TOST."""
    def direction(ci):
        lo,hi=ci
        if lo is None or hi is None:return 0
        return 1 if lo>delta else -1 if hi < -delta else 0
    def inside(ci):
        return ci[0] is not None and ci[1] is not None and ci[0]>=-delta and ci[1]<=delta
    if inside(source90) and inside(target90):return 'equivalent'
    a,b=direction(source95),direction(target95)
    if a and b:return 'transfer' if a==b else 'reverse'
    return 'unresolved'

def clear_reversal(source,target,delta):
    """Inclusive >= delta criterion from the user request, exact ties separate."""
    x,y=np.asarray(source),np.asarray(target)
    i,j=np.triu_indices(x.shape[-1],1)
    a,b=x[...,i]-x[...,j],y[...,i]-y[...,j]
    informative=(np.abs(a)>=delta)&(np.abs(b)>=delta)
    reverse=((a>=delta)&(b<=-delta))|((a<=-delta)&(b>=delta))
    total=len(i);count=reverse.sum(-1);n=informative.sum(-1)
    return dict(total_pairs=total,informative_pairs=n,reversed_pairs=count,
        reversal_fraction_total=count/total,
        reversal_fraction_informative=np.divide(count,n,out=np.full(np.shape(n),np.nan),where=n>0))

def decision_regret(source,target):
    """Descriptive test-metric selection, tie order follows registered method IDs."""
    x,y=np.asarray(source),np.asarray(target)
    selected=np.argmax(x,axis=-1);target_selected=np.argmax(y,axis=-1)
    target_at_source=np.take_along_axis(y,np.expand_dims(selected,-1),axis=-1)[...,0]
    target_max=np.max(y,axis=-1)
    source_top=x==np.max(x,axis=-1,keepdims=True);target_top=y==np.max(y,axis=-1,keepdims=True)
    return dict(regret_raw=target_max-target_at_source,
        top1_exact_match=(selected==target_selected).astype(float),
        source_choice_is_target_optimal=(target_at_source==target_max).astype(float),
        fractional_top_set_agreement=(source_top&target_top).sum(-1)/(source_top|target_top).sum(-1),
        source_top_target_optimal_fraction=(source_top&target_top).sum(-1)/source_top.sum(-1),
        source_tied_maxima=(x==np.max(x,axis=-1,keepdims=True)).sum(-1),
        target_tied_maxima=(y==np.max(y,axis=-1,keepdims=True)).sum(-1),
        source_selected_index=selected,target_selected_index=target_selected)

def aligned_users(protocols):
    """Common users when sets differ, with exact full-truth equality on retained users."""
    common=None;counts={}
    for name,(users,truth,_) in protocols.items():
        if len(np.unique(users))!=len(users):raise ValueError('Duplicate user IDs require cluster reduction')
        counts[name]=len(users)
        common=np.asarray(users) if common is None else np.intersect1d(common,users)
    common=np.sort(common)
    if not len(common):raise ValueError('No common evaluation users')
    aligned={};reference=None
    for name,(users,truth,matrix)in protocols.items():
        order=np.argsort(users);ix=order[np.searchsorted(np.asarray(users)[order],common)]
        aligned_truth=np.asarray(truth)[ix]
        if reference is None:reference=aligned_truth
        elif not np.array_equal(reference,aligned_truth):raise ValueError('Full relevant truth differs on common users')
        aligned[name]=np.asarray(matrix)[ix]
    return common,aligned,dict(n_common_users=len(common),original_user_counts=counts,
        excluded_users={name:n-len(common)for name,n in counts.items()},
        user_sets_identical=all(n==len(common)for n in counts.values()),
        full_truth_equal_on_common_users=True)

def input_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
