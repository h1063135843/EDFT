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
from mmseg.datasets.pipelines import ImageSegClassMapping, LoadAnnotations
from mmseg.models import build_segmentor
from mmseg.ops import resize


def _load_points(pts_filename, with_color=False):
    file_client_args = dict(backend='disk')
    file_client = mmcv.FileClient(**file_client_args)
    pts_bytes = file_client.get(pts_filename)
    points = np.frombuffer(pts_bytes, dtype=np.float32)
    points = points.reshape(-1, 6)
    points = torch.tensor(points if with_color else points[:, 0:3])
    return points


def load_annotation2d(ann_filename):
    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)
    results = dict(ann_info=dict(seg_map=ann_filename), seg_fields=[])
    load = LoadAnnotations(reduce_zero_label=False)
    class_mapping = ImageSegClassMapping(VALID_CLASS_IDS, 40)
    results = class_mapping(load(results))

    return results['gt_semantic_seg']


def get_feature_per_image(points, img_feature, depth, depth2img):
    h, w, c = img_feature.shape
    pts_2d = points_cam2img(points, depth2img, True)

    x, y, z = pts_2d[:, 0], pts_2d[:, 1], pts_2d[:, 2]
    xflag = torch.gt(x, 0) & torch.lt(x, w)
    yflag = torch.gt(y, 0) & torch.lt(y, h)
    zflag = torch.gt(z, 0)
    flag = xflag & yflag & zflag

    z = (pts_2d[flag, 2] * 1000).long()
    pts_2d = pts_2d.long()
    x, y = pts_2d[flag, 0], pts_2d[flag, 1]
    raw_z = depth[y, x]
    depth_constrain = torch.gt(z, raw_z - 30) & torch.lt(z, raw_z + 30)
    tmp=flag.clone()
    flag[tmp] = depth_constrain
    x, y = x[depth_constrain], y[depth_constrain]

    pts_img = torch.full([pts_2d.shape[0], c], 0).byte()
    pts_img[flag] = img_feature[y, x]
    return pts_img


rootdir = r'G:\Datasets\scannet'
label2d_dir = osp.join(rootdir, 'nyu_annotations', 'validation')


def get_info(mode):
    if mode == 'train':
        path = osp.join(rootdir, 'scannet_infos_train.pkl')
    elif mode == 'val':
        path = osp.join(rootdir, 'scannet_infos_val.pkl')

    info = mmcv.load(path)
    return info


def get_scannet_list(mode):
    if mode == 'train':
        path = osp.join(rootdir, 'meta_data', 'scannetv2_train.txt')
    elif mode == 'val':
        path = osp.join(rootdir, 'meta_data', 'scannetv2_val.txt')

    scannet_list = dict()
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            scannet_list[line.split()[0]] = i
    return scannet_list


def get_depth2img_by_scene_name(scene, scan_list, info):
    data_info = info[scan_list[scene]]
    pts_path = data_info['pts_path']
    intrinsic = data_info['intrinsics']
    assert scene in pts_path

    depth2img = []
    for i, extrinsic in enumerate(data_info['extrinsics']):
        depth2img.append(intrinsic @ np.linalg.inv(extrinsic))
    return depth2img


def get_points(scene, scan_list, info):
    data_info = info[scan_list[scene]]
    pts_path = data_info['pts_path']
    assert scene in pts_path

    points = _load_points(osp.join(rootdir, pts_path))
    return points


mode = 'val'
global_info = get_info(mode)
global_list = get_scannet_list(mode)
save_path = 'ground'
# save_path = '056800'
mmcv.mkdir_or_exist(osp.join(rootdir, save_path))


# arrays: h,w,c shape list of number_views length
def save_logit_by_one_scene(scene,
                            arrays,
                            filelist,
                            save=True,
                            aggregation='max'):
    points = get_points(scene, global_list, global_info)
    depth2img = get_depth2img_by_scene_name(scene, global_list, global_info)

    pts_imgs = []
    for i, img in enumerate(arrays):
        depth_path = osp.join(rootdir, 'images', 'depth',
                              filelist[i].replace('jpg', 'png'))
        depth = mmcv.imread(depth_path, 'unchanged')
        depth = torch.from_numpy(depth.astype(
            np.float)).unsqueeze(0).unsqueeze(0)
        align_depth = resize(depth, img.shape[:2])
        pts_img = get_feature_per_image(
            points, img, align_depth[0][0],
            torch.tensor(depth2img[i], dtype=torch.float32))
        pts_imgs.append(pts_img)

    points_aug = torch.stack(pts_imgs, dim=0)  # n_view, n_points, c
    if aggregation == 'max':
        points_aug = points_aug.max(0)[0]
    elif aggregation == 'avg':
        points_sum = torch.sum(points_aug, 0, keepdim=False)
        count = torch.count_nonzero(points_aug, 0)
        points_aug = points_sum / count

    if save:
        points = np.concatenate([points, points_aug.numpy()], axis=1)
        points.tofile(osp.join(rootdir, save_path, scene + '.bin'))
    return points_aug


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--mode', help='decided to infer which directory')
    parser.add_argument(
        '--ground', action='store_true', help='whether to use ground truth')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    mode = args.mode
    if mode == 'train':
        cfg.data.test.img_dir = osp.join('images', 'training')
        cur_scene = 'scene0000_00'  # first scene sort by name
        global global_info, global_list
        global_info = get_info(mode)
        global_list = get_scannet_list(mode)
    else:
        cur_scene = 'scene0011_00'
        # cur_scene = 'scene0568_00'

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    torch.cuda.empty_cache()

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    seg_logits = []
    thread_list = []
    ori_filelist = []
    for batch_indices, data in zip(loader_indices, data_loader):
        ori_file = data['img_metas'][0].data[0][0]['ori_filename']
        scene = ori_file[:12]
        if cur_scene != scene:
            tmp = seg_logits.copy()
            tmp2 = ori_filelist.copy()
            seg_logits.clear()
            ori_filelist.clear()

            t = Thread(
                target=save_logit_by_one_scene, args=(cur_scene, tmp, tmp2))
            t.start()
            thread_list.append(t)
            cur_scene = scene
        if args.ground:
            label = load_annotation2d(
                osp.join(label2d_dir, ori_file.replace('.jpg', '.png')))

            h, w = label.shape
            cx, cy = torch.meshgrid(
                torch.tensor(range(h)), torch.tensor(range(w)))
            label = torch.tensor(label).reshape(-1)
            c = label[label < 20].long()
            cx = cx.reshape(-1)[label < 20].long()
            cy = cy.reshape(-1)[label < 20].long()

            seg_logit = torch.zeros((h, w, 20)).byte()
            seg_logit = seg_logit.index_put_((cx, cy, c),
                                             torch.ByteTensor([255]))
            seg_logits.append(seg_logit)
        else:
            with torch.no_grad():
                result, seg_logit = model(return_loss=False, **data)
            seg_logits.append(
                (255 * seg_logit.squeeze(0).permute(1, 2, 0)).byte().cpu())

        ori_filelist.append(ori_file)
        batch_size = len(result) if not args.ground else 1
        for _ in range(batch_size):
            prog_bar.update()

    save_logit_by_one_scene(cur_scene, seg_logits, ori_filelist)

    for t in thread_list:
        t.join()


if __name__ == '__main__':
    main()
