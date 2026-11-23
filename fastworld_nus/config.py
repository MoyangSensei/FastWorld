"""Inference configuration for the nuScenes validation checkpoint."""

CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
POINT_CLOUD_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

MODEL = dict(
    type="FastWorldNuScenes",
    use_grid_mask=True,
    video_test_mode=True,
    use_multi_view=True,
    use_swin=True,
    swin_input_channel=768,
    hidden_channel=256,
    img_backbone=dict(
        type="SwinTransformer3D", arch="tiny", pretrained=None, pretrained2d=True,
        patch_size=(2, 4, 4), window_size=(8, 7, 7), mlp_ratio=4.0, qkv_bias=True,
        qk_scale=None, drop_rate=0.2, attn_drop_rate=0.0, drop_path_rate=0.1, patch_norm=True,
    ),
    pts_bbox_head=dict(
        type="FastWorldNuScenesHead", num_proposals=6, num_views=6, hidden_channel=256,
        num_heads=8, dropout=0.1, use_wm=True, num_traj_modal=3,
    ),
)

PIPELINE = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type="CustomObjectRangeFilter", point_cloud_range=POINT_CLOUD_RANGE),
    dict(type="CustomObjectNameFilter", classes=CLASSES),
    dict(type="NormalizeMultiviewImage", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(
        type="MultiScaleFlipAug3D", img_scale=(1600, 900), pts_scale_ratio=1, flip=False,
        transforms=[
            dict(type="RandomScaleImageMultiViewImage", scales=[0.4]),
            dict(type="PadMultiViewImage", size_divisor=32),
            dict(type="CustomDefaultFormatBundle3D", class_names=CLASSES, with_label=False, with_ego=True),
            dict(type="CustomCollect3D", keys=[
                "gt_bboxes_3d", "gt_labels_3d", "img", "fut_valid_flag", "ego_his_trajs",
                "ego_fut_trajs", "ego_fut_masks", "ego_fut_cmd", "ego_lcf_feat", "gt_attr_labels",
            ]),
        ],
    ),
]

DATASET = dict(
    type="FastWorldNuScenesValidation",
    pipeline=PIPELINE,
    classes=CLASSES,
    modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
    test_mode=True, box_type_3d="LiDAR", queue_length=3, interval_2frames=True,
    pc_range=POINT_CLOUD_RANGE, bev_size=(100, 100),
    map_classes=["divider", "ped_crossing", "boundary"], map_fixed_ptsnum_per_line=20,
    map_eval_use_same_gt_sample_num_flag=True,
)
