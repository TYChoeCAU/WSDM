"""Published S2 frozen U/W construction."""
import numpy as np

def _s2_uw(model,cf,bundle,degree,seed,protocol,dataset):
    import torch
    from diagnostics.revision_evidence.pipeline import sample_unique_negatives,group_rng
    history,valid,test,_=cf.sets_and_degree(bundle);adapter=cf.CFAdapter(model);users=sorted(test);all_s=[];all_i=[]
    weights=degree[1:].numpy().astype(np.float64)+1 if protocol=='W' else None
    if weights is not None:weights/=weights.sum()
    for first in range(0,len(users),256):
        group=users[first:first+256];pools=[]
        for u in group:
            neg,_=sample_unique_negatives(group_rng(dataset,seed,u,'U_control' if protocol=='U' else 'P_control'),bundle['item_num'],100*len(test[u]),history.get(u,set())|valid.get(u,set())|test[u]|{0},weights)
            pools.append(np.asarray(sorted(test[u]|set(neg.tolist())),dtype=np.int64))
        scores=adapter.score_candidates(np.concatenate([np.full(len(p),u) for p,u in zip(pools,group)]),np.concatenate(pools));cursor=0
        for pool in pools:
            value=scores[cursor:cursor+len(pool)];cursor+=len(pool);ids=torch.from_numpy(pool);order=torch.argsort(value,descending=True,stable=True)[:50]
            s=torch.full((50,),-torch.inf);i=torch.zeros(50,dtype=torch.long);s[:len(order)]=value[order];i[:len(order)]=ids[order];all_s.append(s);all_i.append(i)
    split=cf.pack(users,torch.stack(all_s),torch.stack(all_i),[test[u] for u in users]);split['item_degree']=torch.log1p(degree)[split['candidate_ids'].long()]
    return split

