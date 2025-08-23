_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=3, val_interval=1)

randomness = dict(seed=42, deterministic=True)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        milestones=[90, 120],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=160)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=3))

# codec settings
codec = dict(
    type='DecoupledHeatmapBbox', 
    input_size=(512, 512), 
    heatmap_size=(128, 128), 
    root_type='bbox_center_real', # Chosen because instances have a lot of cropping and occlusions; alternative is 'kpt_center' which averages visible keypoints
    bbox_format = 'x1y1x2y2'
)

# model settings
model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='CustomHead',
        in_channels=480,
        num_keypoints=19,
        gfd_channels=32,
        use_bbox=False,
        coord_att_type='Concatenated', # 'Default', 'Concatenated' 'PreFiLM'
        coupled_heatmap_loss=dict(type='FocalHeatmapLoss', loss_weight=1.0),
        decoupled_heatmap_loss=dict(type='FocalHeatmapLoss', loss_weight=4.0),
        contrastive_loss=dict(
            type='InfoNCELoss', temperature=0.05, loss_weight=1.0),
        bbox_loss=dict( 
            type='IoULoss',
            mode='square', # When squaring the IoU, it is penalized more heavily when the overlap is low.
            eps=1e-16,
            reduction='sum',
            loss_weight=1.0
        ),
            decoder=codec,
        conv_type='1x1Conv', # 1x1Conv, Conv2d, DepthwiseSeparableConvModule, DilatedConv, DeformConv2d, AdaptiveRotatedConv2d
        # bbox_format=codec["bbox_format"]
    ),
    train_cfg=dict(max_train_instances=200),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        shift_heatmap=False,
        align_corners=False))

# base dataset settings
dataset_type = 'PigPoseDataset'
data_mode = 'bottomup'
data_root = 'data/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    # BottomupRandomAffine with shift_factor=0.1 and transform_mode='perspective'
    # This transformation can move (translate/rotate/scale) the image and associated bboxes, and cause some bboxes to fall partially or completely outside the image border.
    # NOTE: I will truncate by myself the results. I mean make the bbox be upto 0 no less. now due to the cordinate scheme of the BottomupRandomAffine I am having negative coords. 
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize',
        input_size=codec['input_size'],
        size_factor=64,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# data loaders
train_dataloader = dict(
    batch_size=20,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='pigpose/pigpose_train.json',
        data_prefix=dict(img='pigpose/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='pigpose/pigpose_val.json',
        data_prefix=dict(img='pigpose/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'pigpose/pigpose_val.json',
    nms_thr=0.8,
    score_mode='keypoint',
)

test_evaluator = val_evaluator
