"""Original native Eq12 for ragged pre-Top50 pools, identical to completed study.
Only output allocation/stacking differ from the 50-column native scorer.
"""
import numpy as np,torch
from graph_rerank.reranker import GraphReranker
from studies.s3.cross_common import deadline
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
            row=torch.zeros(n);row[:n]=torch.from_numpy(values[cursor:cursor+n]).float()
            if first==0 and local<3 and n:
                g,*_=direct._graph_scores(r['user'],r['items'].long().to(adapter.device),r['base_score_raw'].to(adapter.device))
                expected=r['base_score_raw']+alpha*(g.cpu()-r['base_score_raw'])
                maximum=max(maximum,float((expected-row[:n]).abs().max()))
                if maximum>1e-3:raise RuntimeError('batched native NonParamGC parity failed')
            out.append(row);cursor+=n
    return out,maximum
