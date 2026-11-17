"""PRM benchmark variant ported from the public TensorFlow reference."""

from __future__ import annotations
import torch
from torch import nn


def masked_softmax(logits, mask):
    values=logits.masked_fill(~mask,-torch.inf)
    probs=torch.softmax(values,dim=-1)
    return torch.where(mask,probs,torch.zeros_like(probs))


class PRM(nn.Module):
    """One-block personalized Transformer reranker.

    Current supported fields are the existing user ID, item ID, frozen
    backbone score, and initial rank position.  User/item embeddings are
    trainable PRM parameters.  Labels never enter forward().
    """
    def __init__(self,user_num,item_num,max_len=50,embedding_dim=16,d_model=64,d_inner=128,n_head=1,dropout=0.2):
        super().__init__(); self.max_len=max_len
        self.user_embedding=nn.Embedding(user_num,embedding_dim,padding_idx=0)
        self.item_embedding=nn.Embedding(item_num+1,embedding_dim,padding_idx=item_num)
        self.input_projection=nn.Linear(embedding_dim*2+1,d_model)
        self.position_embedding=nn.Embedding(max_len,d_model)
        self.encoder=nn.TransformerEncoderLayer(d_model,n_head,d_inner,dropout,activation="relu",batch_first=True,norm_first=False)
        self.batch_norm=nn.BatchNorm1d(d_model)
        self.fc1=nn.Linear(d_model,d_model); self.dropout=nn.Dropout(dropout); self.fc2=nn.Linear(d_model,1)

    def base_tokens(self,batch):
        users=self.user_embedding(batch["users"]).unsqueeze(1).expand(-1,batch["candidate_items"].shape[1],-1)
        items=self.item_embedding(batch["candidate_items"])
        score=batch["score_standardized"].unsqueeze(-1)
        return self.input_projection(torch.cat((users,items,score),dim=-1))

    def forward_from_tokens(self,tokens,positions,mask):
        x=tokens+self.position_embedding(positions.clamp(0,self.max_len-1))
        x=self.encoder(x,src_key_padding_mask=~mask)
        x=x*mask.unsqueeze(-1)
        flat=self.batch_norm(x.reshape(-1,x.shape[-1])).reshape_as(x)
        logits=self.fc2(self.dropout(torch.relu(self.fc1(flat)))).squeeze(-1)
        return masked_softmax(logits,mask)

    def forward(self,batch,positions=None):
        mask=batch["candidate_mask"]
        if positions is None: positions=torch.arange(mask.shape[1],device=mask.device).unsqueeze(0).expand_as(mask)
        return self.forward_from_tokens(self.base_tokens(batch),positions,mask)
