_base_ = [
    'bucketed_co_deformable_detr_r50_1x_coco_iou_lr_2e-4_step_10_11_divide_5_no_self_weight_fixed_seed_1187060654_16072024.py'
]
pretrained = 'models/swin_large_patch4_window12_384_22k.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        out_indices=(1, 2, 3),
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[192*2, 192*4, 192*8]))

# optimizer
optimizer = dict(lr=2e-4,weight_decay=0.05)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
