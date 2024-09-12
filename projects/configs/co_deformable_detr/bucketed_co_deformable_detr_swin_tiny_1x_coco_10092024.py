_base_ = [
    'bucketed_co_deformable_detr_r50_1x_coco_iou_lr_2e-4_step_10_11_divide_5_fixed_seed_1187060654_15072024.py'
]
pretrained = 'models/swin_tiny_patch4_window7_224.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        out_indices=(1, 2, 3),
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[96*2, 96*4, 96*8]))

# optimizer
optimizer = dict(weight_decay=0.05)