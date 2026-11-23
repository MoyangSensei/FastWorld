"""Standalone inference core: no ancestor checkpoints or training caches required."""
import torch
from torch import nn
from fastworld.runtime.decision_utility_world_fastworld import DecisionUtilityWorldFastWorld
from fastworld.runtime.release_graph_switch_fastworld import GraphSafeSelectiveFinalBEVFastWorld

class NativeReleaseScoreFastWorld(nn.Module):
    def __init__(self,source):
        super().__init__();self.double_score=source.double_score
        if self.double_score:raise ValueError('Release requires single native score head')
        self.pred_score=source.pred_score
    def forward(self,proposals,feature):
        b,k,t,_=proposals.shape;pooled=feature.reshape(b,k,t,-1).amax(-2)
        return self.pred_score(pooled).reshape(b,k,6),None,None,None,None,None,None

class EligibleNativeScoreFastWorld(nn.Module):
    def __init__(self,source):
        super().__init__();self.source=source;self.double_score=False;self.eligible=None
        assert not source.double_score
    def forward(self,proposals,feature):
        assert self.eligible is not None
        score,second,*rest=self.source(proposals,feature);assert second is None
        masked=score[...,-1:].masked_fill(~self.eligible[...,None],float('-inf'))
        return torch.cat([score[...,:5],masked],-1),None,*rest

class SelectiveDecisionPlannerFastWorld(nn.Module):
    def __init__(self,body,head,switch):
        super().__init__();self.body=body;self.head=head
        object.__setattr__(self,'last_update_switch',switch)
    def forward(self,features):
        self.last_update_switch.calls=0
        ego=features['ego_status'][:,-1]
        image=self.body._backbone(features['camera_feature'],img_metas=features)
        candidate=self.body.hist_encoding(ego)[:,None]+self.body.init_feature.weight[None]
        proposals=[]
        for refiner in self.body._trajectory_head:candidate,proposals=refiner(candidate,proposals,image)
        assert self.last_update_switch.calls==4
        return self.head(candidate.reshape(len(ego),64,8,256),proposals[-1],ego)
    def close(self):pass

def build_release_core_fastworld(config):
    from fastworld.models.body import PlannerBodyFastWorld
    from fastworld.models.bevformer.batch_attention_fastworld import use_batch_independent_attention_fastworld

    body = PlannerBodyFastWorld(config)
    use_batch_independent_attention_fastworld(body)
    native = NativeReleaseScoreFastWorld(body.scorer)
    del body.scorer

    eligible = EligibleNativeScoreFastWorld(native)
    router = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 256))
    head = DecisionUtilityWorldFastWorld(eligible)
    refiner = body._trajectory_head[0]
    refiner.Bev_refiner = GraphSafeSelectiveFinalBEVFastWorld(refiner.Bev_refiner, native, eligible, router)
    return SelectiveDecisionPlannerFastWorld(body, head, refiner.Bev_refiner)
