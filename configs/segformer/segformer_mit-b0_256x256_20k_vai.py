_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/Vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='pretrain/mit_b0.pth', decode_head=dict(num_classes=6))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
evaluation = dict(metric=['mIoU','mFscore'],save_best='mIoU')