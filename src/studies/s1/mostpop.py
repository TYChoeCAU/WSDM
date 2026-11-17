"""Independent MostPop retrieval, using the published diagnostic implementation."""
import hashlib
import numpy as np
from . import static as aux

def digest_array(a): return hashlib.sha256(np.asarray(a,dtype=np.int64).tobytes()).hexdigest()


def top_ids(items,degree):
    items=np.asarray(items,dtype=np.int64)
    return items[np.lexsort((items,-degree[items]))[:50]]


class Metrics:
    def __init__(self):
        self.ids=[];self.truth=[];self.count=[];self.users=[];self.eligible=[];self.eligible_truth=[]
    def add(self,user,selected,positive,positive_count,eligible=None,eligible_truth=None):
        self.users.append(user);self.ids.append(np.asarray(selected,dtype=np.int64))
        self.truth.append(set(map(int,positive)));self.count.append(positive_count)
        if eligible is not None:self.eligible.append(eligible);self.eligible_truth.append(eligible_truth)
    def finish(self,degree):
        n=len(self.ids);ids=np.zeros((n,50),np.int64);valid=np.zeros((n,50),bool);labels=np.zeros((n,50),bool)
        for i,v in enumerate(self.ids):
            ids[i,:len(v)]=v;valid[i,:len(v)]=True;labels[i,:len(v)]=[int(x) in self.truth[i] for x in v]
        split=dict(candidate_ids=ids,valid_mask=valid,labels=labels,positive_count=np.asarray(self.count))
        values=aux.metric_vectors(degree[ids].astype(np.float64),split);oracle,recall=aux.oracle_vectors(split)
        result=dict(users=n,unique_users=len(set(self.users)),ndcg20=float(values['ndcg@20'].mean()),recall20=float(values['recall@20'].mean()),candidate_recall50=float(recall.mean()),oracle_ndcg20=float(oracle.mean()),selected_top50_ids_sha256=digest_array(ids),full_truth_count=int(sum(self.count)),metric_evaluator='w2w3_auxiliary.metric_vectors -> p0_extract.metric_vectors')
        if self.eligible:
            eligible=np.asarray(self.eligible);retained=np.asarray(self.eligible_truth);count=np.asarray(self.count)
            random_probability=np.minimum(50,eligible)/eligible
            result.update(eligible_catalog_mean=float(eligible.mean()),eligible_catalog_min=int(eligible.min()),random_top50_expected_candidate_recall50=float(np.mean(random_probability*retained/count)),random_top50_requested_50_over_eligible_mean=float(np.mean(50/eligible)),users_with_excluded_relevant_items=int(np.sum(retained<count)),users_with_less_than50_eligible=int(np.sum(eligible<50)))
        return result


def full_catalog(records,degree):
    order=np.arange(1,len(degree),dtype=np.int64);order=order[np.lexsort((order,-degree[order]))]
    result=Metrics()
    for u,positive,count,exclude in records:
        selected=[]
        for item in order:
            if int(item) not in exclude:selected.append(int(item))
            if len(selected)==50:break
        eligible=len(degree)-1-sum(0<i<len(degree) for i in exclude)
        result.add(u,selected,positive,count,eligible,len(set(positive)-exclude))
    return result.finish(degree)

