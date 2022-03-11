import argparse
import mmcv
import numpy as np
import os
import os.path as osp
import sys
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.core.bbox.structures import points_cam2img
from threading import Thread

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

rootdir = r'G:\Datasets\scannet'


def get_info(mode):
    if mode == 'train':
        path = osp.join(rootdir, 'scannet_infos_train.pkl')
    elif mode == 'val':
        path = osp.join(rootdir, 'scannet_infos_val.pkl')

    info = mmcv.load(path)
    return info


def save_info(info, prefix, mode):
    if mode == 'train':
        path = osp.join(rootdir, prefix, 'scannet_infos_train.pkl')
    elif mode == 'val':
        path = osp.join(rootdir, prefix, 'scannet_infos_val.pkl')

    mmcv.dump(info, path, 'pkl')


def change_prefix(prefix, mode):
    info = get_info(mode)
    for i in info:
        i['point_cloud']['num_features'] = 23
        i['pts_path'] = i['pts_path'].replace('points', prefix)
    save_info(info, prefix, mode)

def set_one_scene(prefix):
    info = mmcv.load(osp.join(rootdir, prefix, 'scannet_infos_val.pkl'))
    info=[info[0]]
    mmcv.dump(info, osp.join(rootdir, prefix, '056800.pkl'), 'pkl')

def write_xyz_points(array, filename):
    with open(filename, 'w') as f:
        for row in array:
            f.write(' '.join(map(str, row)))
            f.write('\n')


def _load_points(pts_filename, dim=6):
    file_client_args = dict(backend='disk')
    file_client = mmcv.FileClient(**file_client_args)
    pts_bytes = file_client.get(pts_filename)
    points = np.frombuffer(pts_bytes, dtype=np.float32)
    points = points.reshape(-1, dim)

    points = torch.tensor(points)
    return points[:, :3], points[:, 3:]


def visualize_points(points, label, save=True, save_name='ann.xyz'):
    CLASSES = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
        'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
        'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
        'otherfurniture'
    ]
    PALETTE = np.array([[174, 199, 232], [152, 223, 138], [31, 119, 180],
                        [255, 187, 120], [188, 189, 34], [140, 86, 75],
                        [255, 152, 150], [214, 39, 40], [197, 176, 213],
                        [148, 103, 189], [196, 156, 148], [23, 190, 207],
                        [247, 182, 210], [219, 219, 141], [255, 127, 14],
                        [158, 218, 229], [44, 160, 44], [112, 128, 144],
                        [227, 119, 194], [82, 84, 163], [0, 0, 0]])
    pred_seg_color = PALETTE[label]
    pred_seg_color = np.concatenate([points[:, :3], pred_seg_color], axis=1)
    if save:
        write_xyz_points(pred_seg_color, save_name)


def test_visualize_2d_logits(path):
    path = osp.join(rootdir, path)
    points, label = _load_points(path, 23)
    # from torch.nn import functional as F
    # label = F.softmax(label, dim=1)
    label = label.argmax(dim=1)
    visualize_points(points, label)


test_visualize_2d_logits('segformer_b0_points/scene0568_00.bin')
# change_prefix('segformer_b0_points', 'train')
# change_prefix('segformer_b0_points', 'val')
# set_one_scene('segformer_b0_points')
