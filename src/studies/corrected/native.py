"""Original native Eq12 scoring and candidate helpers."""
import hashlib, json
import numpy as np
import torch
from graph_rerank.graph import NeighborTable
from graph_rerank.reranker import GraphReranker
from baselines.dnr.backbone_adapter import file_sha256 as sha
def deadline():
    return None

def tensor_digest(values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        v=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def phase_data(adapter,phase):
    data=getattr(adapter,phase+'_data')._dataset
    f=data.inter_feat
    return {'user_ids':f[data.uid_field].long().cpu(),'item_ids':f[data.iid_field].long().cpu(),
            'labels':f[data.label_field].float().cpu()}


def verify_split_membership(actual,prior):
    result={}
    for name,values in actual.items():
        expected=prior[name]
        ordered=all(torch.equal(values[k],expected[k]) for k in values)
        def canonical(x):
            index=np.lexsort((x['labels'].numpy(),x['item_ids'].numpy(),x['user_ids'].numpy()))
            return {key:value[index] for key,value in x.items()}
        left,right=canonical(values),canonical(expected)
        same=all(torch.equal(left[k],right[k]) for k in left)
        if not same or (name=='train' and not ordered):
            raise RuntimeError('historical split membership or original train order changed: '+name)
        result[name]={'ordered_equal':ordered,'row_multiset_equal':same,'rows':len(values['labels']),
                      'loader_order_note':'held-out sampler sorts rows by user after dataset.build; train order remains exact'}
    return result


def tables(path,fingerprint,k,family):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata'].item()))
        if meta['fingerprint']!=fingerprint or meta['scorer_config']['k']!=k or meta['scorer_config']['family']!=family:
            raise RuntimeError('graph lineage mismatch: '+str(path))
        data=(NeighborTable(z['user_ids'].copy(),z['user_weights'].copy()),NeighborTable(z['item_ids'].copy(),z['item_weights'].copy()))
    return data,{'path':str(path),'sha256':sha(path),'metadata':meta}


def score_records(adapter,records):
    from .corrected_sampling import rank_record
    result=[]
    for start in range(0,len(records),128):
        deadline();group=records[start:start+128]
        lengths=[len(r['items']) for r in group]
        items=torch.cat([torch.as_tensor(r['items']).long() for r in group])
        users=np.concatenate([np.full(n,r['user'],dtype=np.int64) for r,n in zip(group,lengths)])
        score=adapter.score_candidates(users,items)
        cursor=0
        for i,(r,n) in enumerate(zip(group,lengths)):
            result.append(rank_record(r,score[cursor:cursor+n],top_m=50));cursor+=n
            records[start+i]=None
    return result


def eval_batches(records,phase):
    out=[]
    for start in range(0,len(records),256):
        groups=records[start:start+256];pu=[];pi=[]
        for i,r in enumerate(groups):
            positives=torch.as_tensor(r['positive_items']).long().tolist()
            pu.extend([i]*len(positives));pi.extend(positives)
        out.append({'split':phase,'groups':groups,'positive_u':torch.tensor(pu),'positive_i':torch.tensor(pi)})
    return out


@torch.inference_mode()
def native_scores(adapter,neighbors,records,alpha):
    """Batch the existing GraphReranker Eq12 computation, with direct parity checks."""
    ut,it=neighbors;out=[];maximum=0.0
    direct=GraphReranker(adapter.model,adapter.dataset,ut,it,scorer='nonparamgc',controller='uniform',native_eq12=True,alpha=alpha)
    for first in range(0,len(records),128):
        deadline();group=records[first:first+128];lengths=[len(r['items']) for r in group]
        ui=np.concatenate([np.full(n,r['user'],dtype=np.int64) for r,n in zip(group,lengths)])
        ii=np.concatenate([torch.as_tensor(r['items']).numpy().astype(np.int64) for r in group])
        base=torch.cat([r['base_score_raw'] for r in group]).double().numpy()
        if ut.ids.shape[1]==1:values=base.copy()
        else:
            uid=ut.ids[ui];iid=it.ids[ii];q=ut.weights[ui,:,None]*it.weights[ii,None,:]
            u=np.broadcast_to(uid[:,:,None],q.shape);i=np.broadcast_to(iid[:,None,:],q.shape)
            valid=(u>=0)&(i>=0)&(q>0);weight=q.copy();flat=weight.reshape(len(ii),-1)
            if len(ii):flat[:,0]=(1-flat[:,1:]).sum(1)
            active=valid&(weight>0);pred=np.zeros_like(q)
            if active.any():pred[active]=adapter.score_candidates(u[active].copy(),i[active].copy()).double().numpy()
            weighted=np.where(active,weight,0);den=weighted.sum((1,2));has=den>0
            graph=base.copy();graph[has]=(weighted*pred).sum((1,2))[has]/den[has]
            values=base+alpha*(graph-base)
        cursor=0
        for local,(r,n) in enumerate(zip(group,lengths)):
            row=torch.zeros(50);row[:n]=torch.from_numpy(values[cursor:cursor+n]).float()
            if first==0 and local<3 and n:
                g,*_=direct._graph_scores(r['user'],r['items'].long().to(adapter.device),r['base_score_raw'].to(adapter.device))
                expected=r['base_score_raw']+alpha*(g.cpu()-r['base_score_raw'])
                maximum=max(maximum,float((expected-row[:n]).abs().max()))
                if maximum>1e-3:raise RuntimeError('batched native NonParamGC parity failed')
            out.append(row);cursor+=n
    return torch.stack(out),maximum

