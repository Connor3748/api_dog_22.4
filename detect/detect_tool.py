# Copyright (c) OpenMMLab. All rights reserved.
# this is for crop_dog_face using cpu
import argparse
import os
import os.path as osp
from glob import glob

import cv2
import mmcv
import numpy as np

#
# def import_module_error_func(module_name):
#     """When a function is imported incorrectly due to a missing module, raise
#     an import error when the function is called."""
#
#     def decorate(func):
#         def new_func(*args, **kwargs):
#             raise ImportError(
#                 f'Please install {module_name} to use {func.__name__}. '
#                 'For OpenMMLAB codebases, you may need to install mmcv-full '
#                 'first before you install the particular codebase. ')
#
#         return new_func
#
#     return decorate


# sys.path.append("../")
# try:
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from mmpose.apis import _inference_single_pose_model
# except (ImportError, ModuleNotFoundError):
#     @import_module_error_func('mmdet')
#     def inference_detector(*args, **kwargs):
#         pass
#
#
#     @import_module_error_func('mmdet')
#     def init_detector(*args, **kwargs):
#         pass
#
#
#     @import_module_error_func('mmpose')
#     def init_pose_model(*args, **kwargs):
#         pass
#
#
#     @import_module_error_func('mmpose')
#     def inference_top_down_pose_model(*args, **kwargs):
#         pass
#
#
#     @import_module_error_func('mmpose')
#     def vis_pose_result(*args, **kwargs):
#         pass
try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')


class DogFaceDetect():
    def __init__(self, args=None):
        if args is None:
            parser = argparse.ArgumentParser(description='MMAction2 demo')
            parser.add_argument('--config',
                                default=('slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'),
                                help='skeleton model config file path')
            parser.add_argument('--data_path', default=osp.join('..', 'dataset', 'dog'), help='data file path')
            parser.add_argument('--det-config', default='yolox_x_8x8_300e_coco.py',
                                help='Dog detection config file path (from mmdet)')
            parser.add_argument('--det-checkpoint',
                                default='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
                                help='dog detection checkpoint file/url')
            parser.add_argument(
                '--pose-config',
                default='hrnet_w32_animalpose_256x256.py',
                help='dog pose estimation config file path (from mmpose)')
            parser.add_argument(
                '--pose-checkpoint',
                default='https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth'
                , help='dog pose estimation checkpoint file/url')
            parser.add_argument('--det-score-thr', type=float, default=0.4, help='the threshold of dog detection score')
            parser.add_argument('--device', type=str, default='cuda:2', help='CPU/CUDA device option')
            parser.add_argument('--dog_cat', default='dog', help='dog or cat')
            parser.add_argument('--output_size', type=int, default=256, help='img_size')
            args = parser.parse_args()
        self.args = args
        self.detect_model = init_detector(args.det_config, args.det_checkpoint, args.device)
        det_label = 16 if args.dog_cat == 'dog' else 15
        assert self.detect_model.CLASSES[det_label] == args.dog_cat, (
            'We require you to use a detector ''trained on COCO')
        self.img_size = args.output_size
        self.paths = glob(osp.join(args.data_path, 'Validation', '*', '**')) + glob(
            osp.join(args.data_path, 'Training', '*', '**'))

        self.pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
        self.checkfile = list()

    def img2cropface(self, img_path):
        det_result = inference_detector(self.detect_model, img_path)[16]
        dataname, label = img_path.split('/')[-2], labelfind(img_path)

        det_check = det_result[:, 4] >= self.args.det_score_thr
        if any(det_check):
            det_result = det_result[det_check]
            det_result = [dict(bbox=x) for x in list(det_result)]
            poses, _ = _inference_single_pose_model(self.pose_model, img_path, bbox_xywh)
            pose = inference_top_down_pose_model(self.pose_model, img_path, det_result, format='xyxy')[0]
            left_right_eyes = pose[0]['keypoints'][0, 0] - pose[0]['keypoints'][1, 0]
            nose_between_eyes = pose[0]['keypoints'][1, 0] < pose[0]['keypoints'][4, 0] < pose[0]['keypoints'][0, 0]
            eye_conf = all(pose[0]['keypoints'][0:2, 2] > 0.7)

            if eye_conf and left_right_eyes > 0 and nose_between_eyes:
                img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = facecrop(pose, img_raw, (self.img_size, self.img_size))
                count = count + 1 if dataname in checkfile else 0
                name = osp.join(frame_path.split('/')[-3].lower()[0:3], dataname + f'__{label}_{count}')
                cv2.imwrite(osp.join('cropdata', 'final9', f'{name}.jpg'), img)
                checkfile.append(dataname)
                print((100 * (len(checkfile) - 2) / batch), '% pose detect')


def data_inbalance(datatype, Dtypes, label):
    jump = 8 if not label == 1 else 12
    if label == 2:
        jump = 1 if Dtypes[-2] != datatype else 2
    elif label == 3:
        jump = 0 if Dtypes[-2] != datatype else 1
    elif Dtypes[-2] == datatype:
        jump = jump * 4
    elif Dtypes[-1] == datatype:
        jump = jump * 2
    return jump


def labelfind(image_path):
    import codecs
    import json
    diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']
    label_path = os.path.join('../dataset/dog', image_path.split('/')[-3], image_path.split('/')[-2] + '.json')
    if os.path.exists(label_path) and os.path.exists(image_path):
        f = codecs.open(label_path, 'r')
        data = json.load(f)
        label = diction.index(data['metadata']['inspect']['emotion'])
        return label


def facecrop(p, img_raw, img_size):
    (rx, ry), (lx, ly) = p[0]['keypoints'][0:2, 0:2]
    center = ((lx + rx) // 2, (ly + ry) // 2)
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    scale = 61.44 / (np.sqrt(((rx - lx) ** 2) + ((ry - ly) ** 2)))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += (img_size[0] / 2 - center[0])
    M[1, 2] += (img_size[1] / 2 + int(img_size[1] / 20) - center[1])
    img = cv2.warpAffine(img_raw, M, img_size, borderValue=0.0)
    return img


def main():
    from glob import glob
    api_test = DogFaceDetect()
    data = glob('../../test-dog/cropdata/final6/testset/*')
    for img_path in data:
        api_test.img2cropface(img_path)


if __name__ == '__main__':
    main()
