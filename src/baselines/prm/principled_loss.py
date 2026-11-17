"""Full PRM_O6 convergence plus adversarial consistency objective."""

import torch
import torch.nn.functional as F


def masked_logloss(pred,labels,mask):
    values=F.binary_cross_entropy(pred.clamp(1e-7,1-1e-7),labels,reduction="none")
    return (values*mask).sum()/mask.sum().clamp_min(1)


def item_ranks(scores,mask):
    order=torch.argsort(scores.masked_fill(~mask,-torch.inf),dim=1,descending=True,stable=True)
    ranks=torch.empty_like(order); base=torch.arange(scores.shape[1],device=scores.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(1,order,base); return torch.where(mask,ranks,torch.zeros_like(ranks))


def adjacent_perturbation(scores,mask,generator=None):
    ranks=item_ranks(scores,mask); out=ranks.clone()
    for row in range(len(scores)):
        length=int(mask[row].sum())
        if length<2: continue
        first=int(torch.randint(length-1,(1,),generator=generator).item())
        by_rank=torch.argsort(ranks[row,:length],stable=True); ia=int(by_rank[first]); ib=int(by_rank[first+1])
        out[row,ia],out[row,ib]=out[row,ib].clone(),out[row,ia].clone()
    return out


def contrastive_similarity(x1,x2,mask):
    weight=(item_ranks(x2,mask)-item_ranks(x1,mask)).abs().float()
    return (weight*(x1-x2).square()*mask).sum()/mask.sum().clamp_min(1)


def prm_o6_loss(model,batch,generator=None):
    mask=batch["candidate_mask"]; labels=batch["labels"]; tokens=model.base_tokens(batch)
    initial=torch.arange(mask.shape[1],device=mask.device).unsqueeze(0).expand_as(mask)
    y1=model.forward_from_tokens(tokens,initial,mask)
    pos2=item_ranks(y1,mask)
    y2=model.forward_from_tokens(tokens,pos2,mask)
    pos3=adjacent_perturbation(y1.detach(),mask,generator)
    y3=model.forward_from_tokens(tokens,pos3,mask)
    supervised=masked_logloss(y1,labels,mask)+masked_logloss(y2,labels,mask)+masked_logloss(y3,labels,mask)
    consistency=contrastive_similarity(y1,y2,mask)+contrastive_similarity(y1,y3,mask)+contrastive_similarity(y2,y3,mask)
    return supervised+consistency,{"supervised":supervised,"consistency":consistency,"y1":y1,"y2":y2,"y3":y3,"pos2":pos2,"pos3":pos3}
