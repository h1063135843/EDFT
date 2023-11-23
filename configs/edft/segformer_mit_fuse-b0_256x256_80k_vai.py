_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/Vaihingen_ndsm.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[120.476, 81.7993, 81.1927, 30.672],
    std=[54.8465, 39.3214, 37.9183, 38.0866],
    bgr_to_rgb=False,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        type='EDFT',
        backbone="Segformer",
        in_channels=4,
        weight=0.5,
        overlap=True,
        attention_type='dsa-add',
        same_branch=False),
    decode_head=dict(num_classes=6),
    test_cfg=dict(mode='slide',crop_size=(256,256),stride=(171,171)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=750),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=750,
        end=80000,
        by_epoch=False,
    )
]