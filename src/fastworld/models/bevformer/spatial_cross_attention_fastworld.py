"""Per-sample visible-query indexing for independent FastWorld instances only.
The upstream proposal-conditioned attention reuses batch element zero's visible
indices. This copy keeps parameter names and the batch-one path unchanged.
No registry overrides or changes to the original SpatialCrossAttention class.
"""
import torch
from .spatial_cross_attention import SpatialCrossAttention

class SpatialCrossAttentionFastWorld(SpatialCrossAttention):
    @classmethod
    def from_existing(cls, original):
        instance=cls.__new__(cls)
        instance.__dict__=original.__dict__.copy()
        return instance

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        query=query.to(torch.float32)
        key=key.to(torch.float32)
        value=value.to(torch.float32)
       #query_pos=query_pos.to(torch.float32)
        reference_points_cam=reference_points_cam.to(torch.float32)

        with torch.autocast(device_type='cuda', enabled=False):  # Disable autocasting
            if key is None:
                key = query
            if value is None:
                value = key

            if residual is None:
                inp_residual = query
                slots = torch.zeros_like(query)
            if query_pos is not None:
                query = query + query_pos

            bs, num_query, _ = query.size()

            D = reference_points_cam.size(3)
            indexes = []
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = [mask_per_img[j].sum(-1).nonzero().squeeze(-1) for j in range(bs)]
                indexes.append(index_query_per_img)
            max_len = max(len(each) for camera in indexes for each in camera)
            if max_len == 0:
                # No valid BEV queries; skip attention to avoid invalid CUDA launches.
                return inp_residual

            # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
            queries_rebatch = query.new_zeros(
                [bs, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs, self.num_cams, max_len, D, 2])

            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam):
                    index_query_per_img = indexes[i][j]
                    queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                        j, index_query_per_img]

            num_cams, l, bs, embed_dims = key.shape

            key = key.permute(2, 0, 1, 3).reshape(
                bs * self.num_cams, l, self.embed_dims)
            value = value.permute(2, 0, 1, 3).reshape(
                bs * self.num_cams, l, self.embed_dims)

            if self.full_attention:
                queries = self.encoder(tgt=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                       memory=key).view(bs, self.num_cams, max_len, self.embed_dims)

            else:
                queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                                    key=key, value=value,
                                                    reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len,
                                                                                                   D, 2),
                                                    spatial_shapes=spatial_shapes,
                                                    level_start_index=level_start_index).view(bs, self.num_cams, max_len,
                                                                                              self.embed_dims)
            for j in range(bs):
                for i in range(self.num_cams):
                    index_query_per_img = indexes[i][j]
                    slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

            count = bev_mask.sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots = slots / count[..., None]
            slots = self.output_proj(slots)

            res=self.dropout(slots) + inp_residual
            visible_sample = bev_mask.permute(1,0,2,3).reshape(bs,-1).any(-1)
            res=torch.where(visible_sample[:,None,None],res,inp_residual)

        # res=res.to(torch.float16)

        return res


def use_per_sample_spatial_attention_fastworld(model):
    changed=0
    for parent in model.modules():
        for name,child in list(parent.named_children()):
            if type(child) is SpatialCrossAttention:
                parent._modules[name]=SpatialCrossAttentionFastWorld.from_existing(child)
                changed+=1
    return changed
