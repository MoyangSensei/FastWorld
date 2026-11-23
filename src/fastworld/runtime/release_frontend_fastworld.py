"""Move exact2.5x bilinear preprocessing to GPU; native images stay uint8 in RAM.

This is a common inference frontend option, not a new model contribution.
Unsupported shapes/dtypes fall back to the original CPU feature builder.
The normalizer is exhaustively checked over the n/16 pixel alphabet produced
by2.5x bilinear interpolation of uint8 inputs before enabling a GPU route.
"""
import numpy as np
import torch
from torch.nn import functional as F
from fastworld.models.features import FastWorldFeatureBuilder
from navsim.common.dataclasses import SensorConfig

MEAN=np.array([123.675,116.28,103.53],dtype=np.float32)
STD=np.array([58.395,57.12,57.375],dtype=np.float32)
NORMALIZERS={}

def normalizer_fastworld(device):
    key=str(device)
    if key not in NORMALIZERS:
        import mmcv
        values=np.repeat((np.arange(4081,dtype=np.float32)/16)[:,None,None],3,axis=2)
        expected=torch.from_numpy(mmcv.imnormalize(values.copy(),MEAN,STD,to_rgb=True)).to(device)
        source=torch.from_numpy(values).to(device)
        mean=torch.as_tensor(MEAN,device=device);inverse64=torch.as_tensor(1/STD.astype(np.float64),device=device)
        delta=source-mean
        candidates=dict(float32=delta*inverse64.float(),double_inverse=(delta.double()*inverse64).float())
        matches=[mode for mode,value in candidates.items() if torch.equal(value,expected)]
        NORMALIZERS[key]=(matches[0] if matches else None,mean.reshape(1,3,1,1),inverse64.reshape(1,3,1,1))
    return NORMALIZERS[key]

class GPUResNetFeatureBuilderFastWorld(FastWorldFeatureBuilder):
    def get_unique_name(self):return 'fastworld_gpu_resnet_feature'

    def compute_features(self,agent_input):

        cameras=agent_input.cameras[-1]
        views=[cameras.cam_b0,cameras.cam_f0,cameras.cam_l0,cameras.cam_r0]
        shapes=[None if cam.image is None else cam.image.shape for cam in views]
        supported=torch.cuda.is_available() and all(shape is not None for shape in shapes) and len(set(shapes))==1
        supported=supported and all(cam.image.dtype==np.uint8 for cam in views)
        supported=supported and shapes[0][2]==3 and shapes[0][0]%5==shapes[0][1]%5==0
        if not supported:
            self.last_gpu_route=False;return super().compute_features(agent_input)
        device=torch.device('cuda',torch.cuda.current_device());mode,mean,inverse=normalizer_fastworld(device)
        if mode is None:
            self.last_gpu_route=False;return super().compute_features(agent_input)
        h,w,_=shapes[0];scaled_h=int(h*.4);scaled_w=int(w*.4)
        pixels=torch.from_numpy(np.stack([cam.image for cam in views])).to(device).permute(0,3,1,2).contiguous().float()
        image=F.interpolate(pixels,size=(scaled_h,scaled_w),mode='bilinear',align_corners=False)
        delta=image[:,[2,1,0]]-mean
        normalized=delta*inverse.float() if mode=='float32' else (delta.double()*inverse).float()
        padded_h=(scaled_h+31)//32*32;padded_w=(scaled_w+31)//32*32
        normalized=F.pad(normalized,(0,padded_w-scaled_w,0,padded_h-scaled_h),value=0)
        matrices=[];scale=np.eye(4);scale[0,0]=scale[1,1]=.4
        for cam in views:
            rotation=np.linalg.inv(cam.sensor2lidar_rotation)
            translation=cam.sensor2lidar_translation@rotation.T
            transform=np.eye(4);transform[:3,:3]=rotation.T;transform[3,:3]=-translation
            intrinsic=cam.intrinsics;view=np.eye(4);view[:intrinsic.shape[0],:intrinsic.shape[1]]=intrinsic
            matrices.append(scale@(view@transform.T))
        features=dict(camera_feature=normalized,img_shape=torch.FloatTensor([[padded_h,padded_w,3]]*4),lidar2img=torch.FloatTensor(np.stack(matrices)))
        features['ego_status']=torch.stack([torch.cat([torch.tensor(s.ego_pose),torch.tensor(s.ego_velocity),torch.tensor(s.ego_acceleration),torch.tensor(s.driving_command)]) for s in agent_input.ego_statuses]).float()
        self.last_gpu_route=True;self.normalizer_mode=mode
        return features


def current_sensors_fastworld():
    return SensorConfig(cam_f0=[3],cam_b0=[3],cam_l0=[3],cam_r0=[3],cam_l1=[],cam_l2=[],cam_r1=[],cam_r2=[],lidar_pc=[])
