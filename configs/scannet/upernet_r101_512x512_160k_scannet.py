_base_ = ['../_base_/models/upernet_r50.py', './4x4_512x512_20k_scannet.py']

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        num_classes=20,
        loss_decode=dict(class_weight=[
            2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
            4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
            5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
            5.3954206, 4.6971426
        ])),
    # test_cfg=dict(mode='slide',crop_size=(512,512),stride=(341,341)))
    test_cfg=dict(mode='whole'))
