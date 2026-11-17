"""Original arbitrary-width common ranking evaluator."""
import numpy as np,torch
def array(x,dtype=None):
    return np.asarray(x.detach().cpu().numpy()if hasattr(x,'detach')else x,dtype=dtype)


def metric_vectors(scores,ids,mask,labels,positive):
    scores=array(scores,np.float64);ids=array(ids,np.int64);mask=array(mask,bool);labels=array(labels,bool)
    if not np.isfinite(scores[mask]).all():raise ValueError('Nonfinite valid candidate score')
    ordered=np.lexsort((ids,np.where(mask,-scores,np.inf)),axis=1)
    ranked=np.take_along_axis(labels&mask,ordered,axis=1)
    discount=1/np.log2(np.arange(2,22));prefix=np.r_[0,np.cumsum(discount)]
    result={}
    for k in (10,20):
        hits=ranked[:,:k].sum(1);dcg=(ranked[:,:k]*discount[:k]).sum(1)
        result[f'recall@{k}']=np.divide(hits,positive,out=np.zeros(len(positive)),where=positive>0)
        denom=prefix[np.minimum(k,positive)]
        result[f'ndcg@{k}']=np.divide(dcg,denom,out=np.zeros(len(positive)),where=denom>0)
    return result,ordered
