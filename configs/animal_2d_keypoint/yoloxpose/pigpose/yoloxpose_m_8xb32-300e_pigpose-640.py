_base_ = './yoloxpose_s_8xb32-300e_pigpose-640.py'

widen_factor = 0.75
deepen_factor = 0.67
checkpoint = 'https://download.openmmlab.com/mmpose/v1/pretrained_models/' \
             'yolox_m_8x8_300e_coco_20230829.pth'

# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint),
    ),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    head=dict(head_module_cfg=dict(widen_factor=widen_factor)))

# data
input_size = (640, 640)
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# base dataset settings
dataset_type = 'PigPoseDataset'
data_mode = 'bottomup'
data_root = 'data/'

dataset_pigpose = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ann_file='pigpose/pigpose_train.json',
    data_prefix=dict(img='pigpose/'),
    pipeline=train_pipeline_stage1,
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_pigpose)