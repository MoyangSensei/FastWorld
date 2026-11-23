"""Independent proposal encoder: reference points follow interleaved queues."""
import torch
from .encoder import BEVFormerEncoder
class BEVFormerEncoderFastWorld(BEVFormerEncoder):
    @classmethod
    def from_existing(cls, original):
        instance=cls.__new__(cls);instance.__dict__=original.__dict__.copy();return instance
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                ref_2d=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        output = bev_query
        intermediate = []

        bs=bev_query.shape[0]
        len_bev=bev_query.shape[1]

        with torch.autocast(device_type='cuda', enabled=False):  # Disable autocasting
            if ref_2d is not None:
                ref_pos =( ref_2d[:, :, None, :2]+32)/64

                hybird_ref_2d = torch.stack([ref_pos, ref_pos], 1).reshape(bs*2,len_bev,1,2)

                zs = torch.linspace(self.pc_range[2]-self.lidar_height, self.pc_range[5]-self.lidar_height, self.num_points_in_pillar, dtype=torch.float32,
                                    device=ref_2d.device)

                zs = zs[None, None, :, None].repeat(bs, len_bev, 1, 1)

                ref_pos =self.compute_corners(ref_2d.reshape(-1,3)).reshape(-1,len_bev,4,2)

                zs=zs.repeat(1,1,4,1)

                ref_3d = torch.cat([ref_pos.repeat(1, 1, self.num_points_in_pillar, 1), zs], dim=-1).permute(0, 2, 1, 3)

                reference_points = ref_3d.to(torch.float32).clone()

                reference_points = torch.cat(
                    (reference_points, torch.ones_like(reference_points[..., :1])), -1).permute(1, 0, 2, 3)

            else:
                ref_3d = self.ref_3d.repeat(bs, 1, 1, 1)  # 2,4,64,3

                reference_points = self.reference_points.repeat(1, bs, 1, 1)  # 4,2,64,4

                if prev_bev is not None:
                    # prev_bev = prev_bev.permute(1, 0, 2)
                    prev_bev = torch.stack(
                        [prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)

                    ref_2d = self.ref_2d.repeat(bs, 1, 1, 1)
                    shift_ref_2d = ref_2d.clone()
                    shift_ref_2d += shift[:, None, None, :]
                    hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                        bs * 2, len_bev, -1, 2)
                else:
                    hybird_ref_2d = self.hybird_ref_2d.repeat(bs, 1, 1, 1)  # 4,64,1,2

            reference_points_cam, bev_mask = self.point_sampling(
                reference_points,  kwargs['img_metas'])

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
