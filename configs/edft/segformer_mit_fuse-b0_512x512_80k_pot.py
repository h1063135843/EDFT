_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/Potsdam_ndsm.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='pretrain/mit_b0.pth',
    backbone=dict(
        type='EDFT',
        backbone="Segformer",
        in_channels=4,
        weight=0.8,
        overlap=True,
        attention_type='dsa-add',
        same_branch=False),
    decode_head=dict(num_classes=6),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00010,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)