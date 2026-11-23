"""Common deployment backend: remove host-synchronizing visible index lists.

Every camera evaluates the fixed query tensor; camera visibility masks the
independent deformable outputs before their original ordered accumulation.
This backend is available to both public RN34 and FastWorld. It is not a new
world-model contribution and does not alter learned weights or visibility.
"""
import torch
from fastworld.models.bevformer.spatial_cross_attention import SpatialCrossAttention


class FixedQueryCrossBackendFastWorld(SpatialCrossAttention):
    @classmethod
    def from_existing(cls,source):
        assert not source.full_attention
        instance=cls.__new__(cls);instance.__dict__=source.__dict__.copy();return instance

    def forward(self,query,key,value,residual=None,query_pos=None,key_padding_mask=None,
                reference_points=None,spatial_shapes=None,reference_points_cam=None,bev_mask=None,
                level_start_index=None,flag='encoder',**kwargs):
        with torch.autocast(device_type='cuda',enabled=False):
            query=query.float();key=key.float();value=value.float();reference_points_cam=reference_points_cam.float()
            original=query if residual is None else residual
            if query_pos is not None:query=query+query_pos
            b,n,d=query.shape;cameras,length,batch,dimension=key.shape
            assert batch==b and cameras==self.num_cams and dimension==d and key_padding_mask is None
            points=reference_points_cam.shape[3]
            queries=query[:,None].expand(-1,cameras,-1,-1).reshape(b*cameras,n,d)
            refs=reference_points_cam.permute(1,0,2,3,4).reshape(b*cameras,n,points,2)
            flat_key=key.permute(2,0,1,3).reshape(b*cameras,length,d)
            flat_value=value.permute(2,0,1,3).reshape(b*cameras,length,d)
            attended=self.deformable_attention(query=queries,key=flat_key,value=flat_value,
                reference_points=refs,spatial_shapes=spatial_shapes,level_start_index=level_start_index).reshape(b,cameras,n,d)
            visible=(bev_mask.sum(-1)>0).permute(1,0,2)
            slots=torch.zeros_like(query)
            # Preserve the source camera accumulation order, with no GPU→CPU
            # nonzero or variable-length query tensors.
            for camera in range(cameras):
                slots=slots+torch.where(visible[:,camera,:,None],attended[:,camera],torch.zeros_like(attended[:,camera]))
            count=visible.sum(1).clamp_min(1.0)
            result=self.dropout(self.output_proj(slots/count[...,None]))+original
            return torch.where(visible.reshape(b,-1).any(-1)[:,None,None],result,original)


def use_fixed_query_cross_backend_fastworld(model):
    changed=0
    for parent in list(model.modules()):
        for name,child in list(parent.named_children()):
            if isinstance(child,SpatialCrossAttention) and not isinstance(child,FixedQueryCrossBackendFastWorld):
                parent._modules[name]=FixedQueryCrossBackendFastWorld.from_existing(child);changed+=1
    return changed
