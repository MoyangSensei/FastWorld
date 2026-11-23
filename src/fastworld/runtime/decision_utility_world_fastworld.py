"""Compare verified trajectories and compute a bounded utility correction."""
import torch
from torch import nn
from torch.nn import functional as F


class DecisionUtilityWorldFastWorld(nn.Module):
    def __init__(self,native):
        super().__init__()
        self.base_head=native;self.bound=.25
        for p in native.parameters():p.requires_grad_(False)
        self.feature=nn.Linear(256,64)
        self.encoder=nn.Sequential(nn.Linear(150,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,64),nn.LayerNorm(64))
        self.state_head=nn.Linear(64,3)
        self.correction=nn.Sequential(nn.Linear(1110,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,1))
        self.register_buffer('future_time',(torch.arange(1,9,dtype=torch.float32)/8)[None,None,:,None].expand(1,8,8,1).contiguous(),persistent=False)
        self.register_buffer('candidate_role',F.one_hot((torch.arange(16)>=8).long(),2).float()[None],persistent=False)
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
    def forward(self,candidate,proposals,ego):
        b,k,t,d=candidate.shape;assert (k,t,d)==(64,8,256)
        logits,second,*_=self.base_head(proposals,candidate.reshape(b,512,256));assert second is None
        coarse=logits[...,-1].sigmoid();selected=coarse.argsort(dim=1,descending=True,stable=True)[:,:16]
        active=candidate.gather(1,selected[:,:,None,None].expand(-1,-1,8,256))
        poses=proposals.gather(1,selected[:,:,None,None].expand(-1,-1,8,3))
        prior=logits[...,-1].gather(1,selected)
        z=self.feature(active)
        pair=lambda x:torch.stack([x[:,:8],x[:,8:]],3)
        paired=pair(z);paired_pose=pair(poses)
        mean=paired.mean(3);difference=paired[:,:,:,0]-paired[:,:,:,1]
        xy=paired_pose[...,:2];angle=paired_pose[...,2]
        geometry=torch.cat([xy.reshape(b,8,8,4)/32,angle.sin(),angle.cos()],-1)
        time=self.future_time.expand(b,-1,-1,-1)
        confidence=torch.stack([prior[:,:8].sigmoid(),prior[:,8:].sigmoid()],-1)[:,:,None].expand(-1,-1,8,-1)
        context=torch.cat([mean,difference,geometry,time,ego[:,None,None].expand(-1,8,8,-1),confidence],-1)
        memory=self.encoder(context);state=self.state_head(memory)
        signed=torch.cat([state,-state],1)
        own=torch.cat([memory,memory],1);global_context=memory.mean(1,keepdim=True).expand(-1,16,-1,-1)
        world=torch.cat([own,global_context,signed],-1)

        previous=torch.cat([torch.zeros_like(poses[:,:,:1,:2]),poses[:,:,:-1,:2]],2)
        local=torch.cat([poses[...,:2]/32,poses[...,2:3].sin(),poses[...,2:3].cos(),(poses[...,:2]-previous)/16],-1)
        role=self.candidate_role.expand(b,-1,-1)
        inputs=torch.cat([world.reshape(b,16,1048),local.reshape(b,16,48),prior.sigmoid()[...,None],ego[:,None].expand(-1,16,-1),role],-1)
        residual=self.bound*self.correction(inputs).squeeze(-1).tanh()
        refined=prior+residual;score=refined.sigmoid();pdm=score.new_zeros(b,64).scatter(1,selected,score);pick=pdm.argmax(1)
        return dict(proposals=proposals,pdm_score=pdm,base_pdm=coarse,pred_logit=refined,selected_branches=selected,
                    trajectory=proposals[torch.arange(b,device=z.device),pick],future_pair_xy=xy.detach(),future_contrast_state=state,
                    residual_logit=residual,latent_state_budget=64,world_dimension=64)
