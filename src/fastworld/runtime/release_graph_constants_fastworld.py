"""Common graph-safe constant caching; no weight or model-budget changes."""
import torch

from fastworld.models.bevformer.resnet_image_encoder_fastworld import ResNetImgEncoderFastWorld
from fastworld.models.bevformer.encoder import BEVFormerLayer


class GraphSafeImageEncoderFastWorld(ResNetImgEncoderFastWorld):
    def forward(self,img,len_queue=None,**kwargs):
        b,n,c,h,w=img.shape;image=img.reshape(b*n,c,h,w)
        if self.use_grid_mask:image=self.grid_mask(image)
        features=self.img_backbone(image);features=self.img_neck(features[-self.num_outs:]);image=features[-1]
        bn,c,h,w=image.shape;feature=image.view(b,bn//b,c,h,w);batch,cameras,c,h,w=feature.shape
        feature=feature.flatten(3).permute(1,0,3,2)
        if self.use_cams_embeds:feature=feature+self.cams_embeds[:,None,None,:].to(feature.dtype)
        feature=feature+self.level_embeds[None,None,0:1,:].to(feature.dtype)
        key=(h,w,feature.device)
        if key not in self.shape_cache:
            shape=torch.as_tensor([[h,w]],dtype=torch.long,device=feature.device)
            start=torch.cat((shape.new_zeros((1,)),shape.prod(1).cumsum(0)[:-1]));self.shape_cache[key]=(shape,start)
        shape,start=self.shape_cache[key]
        return feature.permute(0,2,1,3),shape,start,kwargs

class GraphSafeBEVLayerFastWorld(BEVFormerLayer):
    def forward(self,query,key=None,value=None,bev_pos=None,query_pos=None,key_pos=None,attn_masks=None,
                query_key_padding_mask=None,key_padding_mask=None,ref_2d=None,ref_3d=None,bev_h=None,bev_w=None,
                reference_points_cam=None,mask=None,spatial_shapes=None,level_start_index=None,prev_bev=None,**kwargs):
        if self.pre_norm or self.operation_order!=('self_attn','norm','cross_attn','norm','ffn','norm') or attn_masks is not None:
            raise ValueError('Unsupported BEV execution schema')
        cache_key=(bev_h,bev_w,query.device)
        if cache_key not in self.shape_cache:self.shape_cache[cache_key]=(torch.tensor([[bev_h,bev_w]],device=query.device),torch.tensor([0],device=query.device))
        self_shape,self_start=self.shape_cache[cache_key]
        query=self.attentions[0](query,prev_bev,prev_bev,None,query_pos=bev_pos,key_pos=bev_pos,attn_mask=None,
            key_padding_mask=query_key_padding_mask,reference_points=ref_2d,spatial_shapes=self_shape,level_start_index=self_start,**kwargs)
        query=self.norms[0](query)
        query=self.attentions[1](query,key,value,None,query_pos=query_pos,key_pos=key_pos,reference_points=ref_3d,
            reference_points_cam=reference_points_cam,mask=mask,attn_mask=None,key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,level_start_index=level_start_index,**kwargs)
        return self.norms[2](self.ffns[0](self.norms[1](query),None))

def copy_class(source,cls):
    instance=cls.__new__(cls);instance.__dict__=source.__dict__.copy();instance.shape_cache={};return instance

def use_graph_safe_constants_fastworld(agent):
    counts=dict(image=0,layer=0)
    for parent in list(agent.modules()):
        for name,child in list(parent.named_children()):
            if isinstance(child,ResNetImgEncoderFastWorld):parent._modules[name]=copy_class(child,GraphSafeImageEncoderFastWorld);counts['image']+=1
            elif type(child) is BEVFormerLayer:parent._modules[name]=copy_class(child,GraphSafeBEVLayerFastWorld);counts['layer']+=1
    if counts != dict(image=1, layer=1):
        raise RuntimeError(f'Unexpected number of CUDA Graph modules: {counts}')
