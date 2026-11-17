"""Verbatim candidate scoring, candidate packing, coverage and per-user metrics."""
import hashlib,json
import numpy as np,torch
from studies.corrected.corrected_sampling import rank_record
from studies.s3.cross_common import identity
def tensor_hash(value):
    a=torch.as_tensor(value).detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def sampled_top50(adapter,records,group_batch=128):
    result=[]
    for first in range(0,len(records),group_batch):
        group=records[first:first+group_batch];lengths=[len(r['items'])for r in group]
        items=torch.cat([torch.as_tensor(r['items']).long()for r in group])
        users=np.concatenate([np.full(n,r['user'],dtype=np.int64)for r,n in zip(group,lengths)])
        scores=adapter.score_candidates(users,items).float()
        if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite P3 sampled score')
        cursor=0
        for record,n in zip(group,lengths):
            result.append(rank_record(record,scores[cursor:cursor+n],top_m=50));cursor+=n
    return result


def split_from_records(records):
    shape=(len(records),50)
    result=dict(candidate_ids=torch.zeros(shape,dtype=torch.int32),valid_mask=torch.zeros(shape,dtype=torch.bool),
        labels=torch.zeros(shape,dtype=torch.bool),s=torch.zeros(shape,dtype=torch.float32),
        positive_count=torch.tensor([r['positive_count']for r in records],dtype=torch.long),
        user_ids=torch.tensor([r['user']for r in records],dtype=torch.long))
    for n,record in enumerate(records):
        size=len(record['items'])
        if size>50:raise ValueError('P3 Top50 overflow')
        result['candidate_ids'][n,:size]=record['items'];result['labels'][n,:size]=torch.as_tensor(record['labels']).bool()
        result['valid_mask'][n,:size]=True;result['s'][n,:size]=record['base_score_raw']
        if int(result['labels'][n].sum())>int(result['positive_count'][n]):raise ValueError('Retained positives exceed full truth')
    result['candidate_digest']=identity({k:tensor_hash(result[k])for k in ('candidate_ids','valid_mask','labels','positive_count','user_ids')})
    return result


def candidate_summary(split,records,item_num):
    count=split['positive_count'].double();hits=(split['labels']&split['valid_mask']).sum(1).double()
    recall=torch.where(count>0,hits/count,torch.zeros_like(count))
    discount=1/torch.log2(torch.arange(2,22,dtype=torch.float64));prefix=torch.cat((torch.zeros(1),discount.cumsum(0)))
    numerator=prefix[hits.long().clamp_max(20)];denominator=prefix[count.long().clamp_max(20)]
    oracle=torch.where(denominator>0,numerator/denominator,torch.zeros_like(count))
    eligible=torch.tensor([item_num-1-len(set(torch.as_tensor(r['excluded_items']).tolist())-{0})for r in records])
    return dict(candidate_recall50=float(recall.mean()),oracle_ndcg20=float(oracle.mean()),
        zero_hit_users=int((hits==0).sum()),users=len(count),eligible_catalog_min=int(eligible.min()),
        eligible_catalog_max=int(eligible.max()),eligible_catalog_mean=float(eligible.float().mean())),recall,oracle
