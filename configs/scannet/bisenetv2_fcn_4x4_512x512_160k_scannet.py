_base_ = [
    '../_base_/models/bisenetv2.py', '../_base_/datasets/scannet_2d.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(decode_head=dict(num_classes=20), test_cfg=dict(mode='whole'))
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)

data = dict(samples_per_gpu=4, workers_per_gpu=4)
evaluation = dict(interval=160000, metric=['mIoU'])
