_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    rpn_head=dict(
        type='RankBasedRPNHead',
        rank_loss_type = 'BucketedRankSort',
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='GIoULoss', reduction='none'),
        head_weight=0.20),
    roi_head=dict(
        type='RankBasedStandardRoIHead',
        bbox_head=dict(
            type='RankBasedShared2FCBBoxHead',
            rank_loss_type = 'BucketedRankSort',
            num_classes=1203,
            reg_decoded_bbox= True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_bbox=dict(type='GIoULoss', reduction='none'),
            loss_cls=dict(use_sigmoid=True),),
        mask_head=dict(type='RankBasedFCNMaskHead', num_classes=1203)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(sampler=dict(type='PseudoSampler')),
    rcnn=dict(sampler=dict(num=1e10)))

# We use score threshold as 0.60 different from 1e-4
# as the default setting for efficiency.
test_cfg = dict(
    rcnn=dict(
        score_thr=0.60,
        # LVIS allows up to 300
        max_per_img=300))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(samples_per_gpu=4,
            workers_per_gpu=4,
            train=dict(dataset=dict(pipeline=train_pipeline)))

optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=0.0001)
