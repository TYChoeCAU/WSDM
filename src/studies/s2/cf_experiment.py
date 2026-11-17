"""Published S2 MF-BPR fitting and frozen evaluation core.
Numerical functions are extracted verbatim; orchestration paths are supplied by run_s2.
"""
from __future__ import annotations
import collections, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from recbole.model.general_recommender.bpr import BPR
from recbole.data.interaction import Interaction
from diagnostics.revision_evidence.common import atomic_json, write_csv, read_csv, seed_everything, stable_order, ranking_metrics
RUN = None
SCOPE = {}
def deadline(): return None
def gate():
    if RUN is None: raise RuntimeError("S2 runtime is not configured")

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def coord_dir(dataset, seed):
    return RUN / "cf" / dataset / f"seed_{seed}"


def split_bundle(dataset, seed):
    return load(RUN / "private/splits" / dataset / f"seed_{seed}.pt")


def by_user(split, positive_only=False):
    out = collections.defaultdict(set)
    labels = split["labels"].numpy()
    for u,i,y in zip(split["user_ids"].tolist(),split["item_ids"].tolist(),labels):
        if u > 0 and i > 0 and (not positive_only or y > 0):
            out[u].add(i)
    return dict(out)


def sets_and_degree(bundle):
    history = by_user(bundle["train"])
    valid = by_user(bundle["valid"], True)
    test = by_user(bundle["test"], True)
    degree = torch.zeros(bundle["item_num"],dtype=torch.float32)
    for items in history.values():
        degree[list(items)] += 1
    return history,valid,test,degree


class ShapeDataset:
    def __init__(self,bundle): self.bundle=bundle
    def num(self,field):
        return self.bundle["user_num"] if field == self.bundle["uid_field"] else self.bundle["item_num"]


def model_for(bundle,device):
    config={"USER_ID_FIELD":bundle["uid_field"],"ITEM_ID_FIELD":bundle["iid_field"],"NEG_PREFIX":"neg_","embedding_size":64,"device":device}
    return BPR(config,ShapeDataset(bundle)).to(device)


def select_top(scores,items,k):
    """Exact score-desc/item-ID-asc selection, with stable boundary ties."""
    k=min(k,scores.shape[1])
    values,index=torch.topk(scores,k,dim=1,sorted=False)
    threshold=values.min(1).values
    ambiguous=(scores==threshold[:,None]).sum(1) > (values==threshold[:,None]).sum(1)
    # Input item order is ascending in each catalog block. Merge inputs are
    # explicitly sorted below, so boundary ties cannot exclude a smaller ID.
    if ambiguous.any():
        rows=torch.where(ambiguous)[0]
        order=stable_order(scores[rows],items[rows])[:,:k]
        index[rows]=order
        values[rows]=torch.gather(scores[rows],1,order)
    ids=torch.gather(items,1,index)
    order=stable_order(values,ids)
    return torch.gather(values,1,order),torch.gather(ids,1,order)


@torch.inference_mode()
def stream_top(model,users,excluded,k=50,user_batch=256,item_chunk=8192,degree=None):
    device=model.user_embedding.weight.device
    n_items=model.item_embedding.num_embeddings
    all_s,all_i=[],[]
    for first in range(0,len(users),user_batch):
        if first % (user_batch*10)==0: deadline()
        batch=users[first:first+user_batch]
        ex=excluded[first:first+user_batch]
        uid=torch.as_tensor(batch,device=device,dtype=torch.long)
        ue=model.user_embedding(uid)
        ex_rows=[];ex_items=[]
        for r,seen in enumerate(ex):
            ex_rows.extend([r]*len(seen));ex_items.extend(sorted(seen))
        er=torch.tensor(ex_rows,device=device,dtype=torch.long)
        ei=torch.tensor(ex_items,device=device,dtype=torch.long)
        best_s=best_i=None
        for start in range(1,n_items,item_chunk):
            end=min(n_items,start+item_chunk)
            ids=torch.arange(start,end,device=device).expand(len(batch),-1)
            score=ue @ model.item_embedding.weight[start:end].T if degree is None else degree[start:end].to(device).expand(len(batch),-1).clone()
            chosen=(ei>=start)&(ei<end)
            score[er[chosen],ei[chosen]-start]=-torch.inf
            val,item=select_top(score,ids,k)
            if best_s is None: best_s,best_i=val,item
            else: best_s,best_i=select_top(torch.cat((best_s,val),1),torch.cat((best_i,item),1),k)
        all_s.append(best_s.cpu());all_i.append(best_i.cpu())
    return torch.cat(all_s),torch.cat(all_i)


def pack(users,scores,items,relevant):
    valid=torch.isfinite(scores)
    ids=items.clone().int();ids[~valid]=0
    values=scores.clone().float();values[~valid]=0
    labels=torch.stack([torch.isin(row.long(),torch.tensor(sorted(pos),dtype=torch.long)) for row,pos in zip(ids,relevant)]) & valid
    return {"user_ids":torch.tensor(users,dtype=torch.int32),"candidate_ids":ids,"s":values,"valid_mask":valid,"labels":labels,"positive_count":torch.tensor([len(x) for x in relevant],dtype=torch.long)}


def eval_catalog(model,bundle,phase):
    history,valid,test,_=sets_and_degree(bundle)
    truth=valid if phase=="valid" else test
    users=sorted(truth)
    exclusions=[history.get(u,set()) | (valid.get(u,set()) if phase=="test" else set()) for u in users]
    if any(truth[u] & seen for u,seen in zip(users,exclusions)):
        raise RuntimeError(f"target/exclusion overlap in {phase}; must audit before evaluation")
    s,i=stream_top(model,users,exclusions)
    split=pack(users,s,i,[truth[u] for u in users])
    metrics=ranking_metrics(split["s"],split)
    metrics["candidate_recall@50"]=float((split["labels"].sum(1)/split["positive_count"].clamp_min(1)).double().mean())
    return metrics,split,exclusions


def model_signature(bundle,seed,profile):
    value={"seed":seed,"profile":profile,"split_digests":bundle["split_digests"],"mapping_digests":bundle["mapping_digests"],"target":"threshold-positive train rows; all observed train excluded from negatives","score":"raw MF dot product"}
    value["sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return value


def trial(dataset,seed,lr,wd,device):
    gate();bundle=split_bundle(dataset,seed)
    profile={**SCOPE["cf_profile_search"],"learning_rate":lr,"actual_weight_decay":wd}
    out=coord_dir(dataset,seed)/f"lr_{lr:g}_wd_{wd:g}"
    signature=model_signature(bundle,seed,profile)
    result=out/"result.json"
    if result.is_file():
        saved=json.loads(result.read_text())
        assert saved["identity"]==signature and (out/"best_checkpoint.pt").is_file()
        return saved
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"fixed_settings.json",signature)
    seed_everything(seed);model=model_for(bundle,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    train=bundle["train"]
    positives=train["labels"]>0
    user=train["user_ids"][positives].long().to(device)
    item=train["item_ids"][positives].long().to(device)
    ni=bundle["item_num"]
    history=(train["user_ids"].long()*ni+train["item_ids"].long()).unique(sorted=True).to(device)
    # Sentinal permits safe searchsorted for every sampled encoded pair.
    history=torch.cat((history,torch.tensor([bundle["user_num"]*ni+ni],device=device)))
    best=-float("inf");best_epoch=0;stale=0;curves=[];start_epoch=1
    started=time.monotonic()
    resume=out/"progress.pt"
    if resume.is_file():
        cp=load(resume);assert cp["identity"]==signature
        model.load_state_dict(cp["state_dict"]);optimizer.load_state_dict(cp["optimizer"])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value): state[key]=value.to(device)
        best,best_epoch,stale,curves,start_epoch=cp["best"],cp["best_epoch"],cp["stale"],cp["curves"],cp["epoch"]+1
    for epoch in range(start_epoch,501):
        deadline();seed_everything(seed*10000+epoch)
        model.train();order=torch.randperm(len(user),device=device);losses=[]
        epoch_start=time.monotonic()
        for start in range(0,len(order),4096):
            ix=order[start:start+4096];u=user[ix];pos=item[ix]
            neg=torch.randint(1,ni,(len(ix),),device=device)
            for attempt in range(10000):
                key=u*ni+neg;bad=history[torch.searchsorted(history,key)]==key
                if not bad.any(): break
                neg[bad]=torch.randint(1,ni,(int(bad.sum()),),device=device)
            else: raise RuntimeError("training user has no eligible negative")
            interaction=Interaction({model.USER_ID:u,model.ITEM_ID:pos,model.NEG_ITEM_ID:neg})
            optimizer.zero_grad(set_to_none=True);loss=model.calculate_loss(interaction);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        row={"dataset":dataset,"backbone":"MF_BPR","seed":seed,"learning_rate":lr,"weight_decay":wd,"epoch":epoch,"train_loss":float(np.mean(losses)),"epoch_seconds":time.monotonic()-epoch_start,"validation_ndcg@20":"","validation_candidate_recall@50":""}
        if epoch%10==0:
            model.eval();val,_,_=eval_catalog(model,bundle,"valid")
            row.update({"validation_ndcg@20":val["ndcg@20"],"validation_candidate_recall@50":val["candidate_recall@50"]})
            if val["ndcg@20"]>best:
                best=val["ndcg@20"];best_epoch=epoch;stale=0
                save(out/"best_checkpoint.pt",{"identity":signature,"state_dict":model.state_dict(),"selected_epoch":epoch,"validation_metrics":val,"config":profile})
            else: stale+=1
            print(json.dumps(row),flush=True)
        curves.append(row)
        write_csv(out/"training_curves.csv",curves)
        if epoch%10==0:
            save(resume,{"identity":signature,"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"best":best,"best_epoch":best_epoch,"stale":stale,"curves":curves})
        if epoch>=50 and stale>=10: break
    payload={"dataset":dataset,"seed":seed,"identity":signature,"status":"BUDGET_LIMITED" if epoch==500 and stale<10 else "EARLY_STOPPED","selected_epoch":best_epoch,"epochs_ran":epoch,"validation_ndcg@20":best,"checkpoint":str(out/"best_checkpoint.pt"),"wall_seconds":time.monotonic()-started,"no_test_used_for_selection":True}
    atomic_json(result,payload)
    return payload


def choose(dataset):
    gate();rows=[]
    for lr in (0.001,0.003):
        for wd in (0.0,0.0001):
            p=coord_dir(dataset,2020)/f"lr_{lr:g}_wd_{wd:g}"/"result.json"
            if not p.is_file(): raise RuntimeError(f"four-profile validation search incomplete: {p}")
            rows.append(json.loads(p.read_text()))
    selected=max(rows,key=lambda x:x["validation_ndcg@20"])
    atomic_json(RUN/"cf"/dataset/"selected_profile.json",{"selection":"seed2020 full catalog validation NDCG@20 only","learning_rate":selected["identity"]["profile"]["learning_rate"],"weight_decay":selected["identity"]["profile"]["actual_weight_decay"],"seed2020_checkpoint":selected["checkpoint"],"validation_ndcg@20":selected["validation_ndcg@20"],"all_trials":rows})


class CFAdapter:
    def __init__(self,model): self.model=model
    @torch.inference_mode()
    def score_candidates(self,users,items):
        device=self.model.user_embedding.weight.device;out=[]
        users=torch.as_tensor(users,dtype=torch.long);items=torch.as_tensor(items,dtype=torch.long)
        for first in range(0,len(users),131072):
            u=users[first:first+131072].to(device);i=items[first:first+131072].to(device)
            out.append((self.model.user_embedding(u)*self.model.item_embedding(i)).sum(1).cpu())
        return torch.cat(out) if out else torch.empty(0)

