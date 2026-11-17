"""Published fresh U/W/F diagnostic functions."""
import json
import numpy as np
from scipy.stats import rankdata

def array(value, dtype=None):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value, dtype=dtype)


def oracle_vectors(split):
    """Full held-out ideal DCG, even when truth lies outside retrieved Top-50."""
    mask, labels = array(split['valid_mask'], bool), array(split['labels'], bool)
    count = array(split['positive_count'], np.int64); hit = (mask & labels).sum(1)
    if np.any(hit > count) or np.any(labels & ~mask): raise ValueError('Invalid full-truth denominator')
    prefix = np.r_[0., np.cumsum(1 / np.log2(np.arange(2, 22)))]
    denom = prefix[np.minimum(count, 20)]
    oracle = np.divide(prefix[np.minimum(hit, 20)], denom, out=np.zeros(len(count)), where=denom > 0)
    recall = np.divide(hit, count, out=np.zeros(len(count)), where=count > 0)
    return oracle, recall


def spearman_rows(scores, popularity, mask):
    """Average-rank Spearman, undefined for short/constant lists."""
    scores, popularity, mask = array(scores, np.float64), array(popularity, np.float64), array(mask, bool)
    if scores.shape != popularity.shape or scores.shape != mask.shape: raise ValueError('List alignment differs')
    out = np.full(len(scores), np.nan)
    for n in np.unique(mask.sum(1)):
        if n < 2: continue
        idx = np.flatnonzero(mask.sum(1) == n)
        left = scores[idx][mask[idx]].reshape(len(idx), n); right = popularity[idx][mask[idx]].reshape(len(idx), n)
        if not np.isfinite(left).all() or not np.isfinite(right).all(): raise ValueError('Nonfinite valid score')
        a = rankdata(left, method='average', axis=1); b = rankdata(right, method='average', axis=1)
        a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
        denominator = np.sqrt((a*a).sum(1) * (b*b).sum(1))
        out[idx] = np.divide((a*b).sum(1), denominator, out=np.full(len(idx), np.nan), where=denominator > 0)
    return out


def summarize_spearman(values, meta):
    valid = values[np.isfinite(values)]
    return dict(**meta, total_users=len(values), defined_users=len(valid), undefined_users=int((~np.isfinite(values)).sum()),
                mean=float(valid.mean()) if len(valid) else None,
                **{f'q{int(q*100):02}': float(np.quantile(valid, q)) if len(valid) else None for q in (.05,.25,.5,.75,.95)},
                undefined_policy='fewer than two valid candidates or a constant score/popularity list, retained in count')


def _catalog_split(adapter,records,history,valid,item_degree):
    import torch
    from diagnostics.score_degree_essentials.full_catalog_eval import merge_top50
    all_s=[];all_i=[];users=[];truths=[]
    for first in range(0,len(records),64):
        group=records[first:first+64];uid=[r[0] for r in group];users.extend(uid);truths.extend([r[2] for r in group])
        best_s=torch.full((len(uid),50),-torch.inf);best_i=torch.full((len(uid),50),len(item_degree),dtype=torch.long)
        exclusions=[history.get(u,set())|valid.get(u,set()) for u in uid]
        for start in range(1,len(item_degree),8192):
            ids=torch.arange(start,min(start+8192,len(item_degree)))
            # Frozen adapters return inference tensors. Clone outside inference
            # mode before in-place exclusion masking, preserving score values.
            values=adapter.score_candidates(torch.tensor(uid).repeat_interleave(len(ids)),ids.repeat(len(uid))).reshape(len(uid),-1).clone()
            for n,seen in enumerate(exclusions):
                if seen:values[n,torch.isin(ids,torch.tensor(sorted(seen)))]=-torch.inf
            best_s,best_i=merge_top50(best_s,best_i,values,ids.expand(len(uid),-1))
        all_s.append(best_s);all_i.append(best_i)
        if first%640==0:print(json.dumps(dict(stage='E_S1_F_frozen_inference',users_done=min(first+64,len(records)),users_total=len(records))),flush=True)
    s,ids=torch.cat(all_s),torch.cat(all_i);mask=torch.isfinite(s);ids=ids.masked_fill(~mask,0);s=s.masked_fill(~mask,0)
    labels=torch.stack([torch.isin(i,t.long()) for i,t in zip(ids,truths)])&mask
    return dict(user_ids=torch.tensor(users),candidate_ids=ids,s=s,labels=labels,valid_mask=mask,
                positive_count=torch.tensor([len(t) for t in truths]),item_degree=torch.log1p(torch.as_tensor(item_degree))[ids])


import torch
def pack(records,degree):
    n=len(records);ids=torch.zeros((n,50),dtype=torch.int32);s=torch.zeros((n,50));mask=torch.zeros((n,50),dtype=torch.bool);labels=torch.zeros_like(mask)
    for j,r in enumerate(records):
        i=torch.as_tensor(r['items']).long();k=len(i);ids[j,:k]=i.int();s[j,:k]=torch.as_tensor(r['base_score_raw']);mask[j,:k]=True
        labels[j,:k]=torch.isin(i,torch.as_tensor(r['positive_items']).long())
    return {'user_ids':torch.tensor([r['user'] for r in records],dtype=torch.int32),'candidate_ids':ids,'s':s,'valid_mask':mask,'labels':labels,
            'positive_count':torch.tensor([r['positive_count'] for r in records]),'item_degree':torch.log1p(torch.as_tensor(degree).float())[ids.long()]}


def metric_vectors(scores, split):
    from common.metrics import metric_vectors as original
    return original(scores, split['candidate_ids'], split['valid_mask'], split['labels'], array(split['positive_count'], np.int64))[0]
