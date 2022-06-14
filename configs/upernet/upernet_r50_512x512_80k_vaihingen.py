_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/vaihingen_GZ.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(171, 171)))