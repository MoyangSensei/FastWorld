import torch
from torch import nn

class GraphSafeSelectiveFinalBEVFastWorld(nn.Module):
    def __init__(self,source,native,eligible_head,router_adapter):
        super().__init__();self.source=source;self.router_adapter=router_adapter
        object.__setattr__(self,'native',native);object.__setattr__(self,'eligible_head',eligible_head)
        self.calls=0;self.selected=None
        self.register_buffer('cached_shapes', torch.tensor([[64, 8]]), persistent=False)
        self.register_buffer('cached_start', torch.tensor([0]), persistent=False)
        assert source.proposal_query and source.bev_h==64 and source.bev_w==8
        assert len(source.bev_decoder.layers)==1
    def forward(self,pose,prev_bev,image_feature):
        self.calls+=1;assert self.calls<=4
        if self.calls<4:return self.source(pose,prev_bev,image_feature)
        b,q,d=prev_bev.shape;assert (q,d)==(512,256)
        adapted=prev_bev.reshape(b,64,8,256)
        adapted=adapted+self.router_adapter(adapted)
        cheap,*_=self.native(pose,adapted.reshape(b,512,256))
        selected=cheap[...,-1].argsort(dim=1,descending=True,stable=True)[:,:16]
        self.selected=selected
        mask=torch.zeros((b,64),device=pose.device,dtype=torch.bool).scatter(1,selected,True)
        self.eligible_head.eligible=mask
        index=(selected[:,:,None]*8+torch.arange(8,device=pose.device)[None,None]).reshape(b,-1)
        gather=lambda value:value.gather(1,index[:,:,None].expand(-1,-1,value.shape[-1]))
        module=self.source;encoder=module.bev_decoder;layer=encoder.layers[0]
        assert layer.operation_order==('self_attn','norm','cross_attn','norm','ffn','norm') and not layer.pre_norm
        assert 'FastWorld' in type(encoder).__name__
        flat,shapes,level_index,kwargs=image_feature
        full_pos=module.positional_encoding(prev_bev.new_zeros((b,64,8))).flatten(2).permute(0,2,1)
        flat_pose=pose.reshape(b,512,3)
        ref=(flat_pose[:,:,None,:2]+32)/64
        hybrid=torch.stack([ref,ref],1).reshape(b*2,512,1,2)
        # Self attention still sees all original512values/grid locations.
        context=layer.attentions[0](prev_bev,None,None,None,query_pos=full_pos,key_pos=full_pos,
            attn_mask=None,key_padding_mask=None,reference_points=hybrid,
            spatial_shapes=self.cached_shapes,level_start_index=self.cached_start,**kwargs)
        query=layer.norms[0](gather(context))
        selected_pose=gather(flat_pose);n=index.shape[1]
        with torch.autocast(device_type='cuda',enabled=False):
            zs=torch.linspace(encoder.pc_range[2]-encoder.lidar_height,encoder.pc_range[5]-encoder.lidar_height,
                              encoder.num_points_in_pillar,dtype=torch.float32,device=pose.device)
            zs=zs[None,None,:,None].repeat(b,n,1,1).repeat(1,1,4,1)
            corners=encoder.compute_corners(selected_pose.reshape(-1,3)).reshape(b,n,4,2)
            ref3=torch.cat([corners.repeat(1,1,encoder.num_points_in_pillar,1),zs],-1).permute(0,2,1,3)
            homogeneous=torch.cat([ref3.float(),torch.ones_like(ref3[...,:1])],-1).permute(1,0,2,3)
            projected,visible=encoder.point_sampling(homogeneous,kwargs['img_metas'])
        verified=layer.attentions[1](query,flat,flat,None,query_pos=None,key_padding_mask=None,
            reference_points=None,spatial_shapes=shapes,reference_points_cam=projected,bev_mask=visible,
            level_start_index=level_index,**kwargs)
        verified=layer.norms[2](layer.ffns[0](layer.norms[1](verified),None))
        return prev_bev.scatter(1,index[:,:,None].expand(-1,-1,256),verified)
