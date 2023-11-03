_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/Vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    decode_head=dict(num_classes=6))


train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
