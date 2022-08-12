# Copyright (c) OpenMMLab. All rights reserved.
# this is for crop_dog_face using cpu
import argparse
import ast
import io
import os
import os.path as osp
from glob import glob
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from torchvision.transforms import transforms

from detect import models
from mmdet.apis import inference_detector, init_detector

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

EYE_FACE_RATIO = 61.44


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


class DogFaceDetect():
    def __init__(self, args=None):
        # det
        self.model_name, self.model_dict, self.models, self.length, self.det_label, self.labels = None, None, None, None, None, list()
        if not args.crop_face:
            self.detect_model, self.pose_model = load_det_model(args)
        self.args = args
        # rec
        self.dog_model_dict, self.dog_model_name, self.dog_models = load_cls_model(
            osp.join(args.cls_check_path, args.dog_path), args.device)
        if args.cat_path == 'cat':
            self.cat_model_dict, self.cat_model_name, self.cat_models = load_cls_model(osp.join(args.cls_check_path, args.cat_path), args.device, 3)
        else :
            self.cat_model_dict, self.cat_model_name, self.cat_models = load_cls_model(
                osp.join(args.cls_check_path, args.cat_path), args.device)
        self.img_size = args.output_size
        self._transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        print('finish_model_load')

    def choice_dog_cat(self, data_type):
        if data_type == 'cat':

            self.model_dict, self.model_name, self.models = self.cat_model_dict, self.cat_model_name, self.cat_models
            self.det_label = 15
            if self.args.cat_path == 'cat':
                self.args.proba_conf = self.args.proba_conf if not self.args.cat_path == 'cat' else [1.4, 0.1, 1.4]
            else:
                self.args.proba_conf = [0.2, 1.4]
        else:
            self.model_dict, self.model_name, self.models = self.dog_model_dict, self.dog_model_name, self.dog_models
            self.det_label = 16
            self.args.proba_conf = self.args.proba_conf if not self.args.dog_path == 'real' else [1.2, 1.4, 0.3]
        self.length = len(self.model_dict)
        assert self.detect_model.CLASSES[self.det_label] == data_type, (
            'We require you to use a detector ''trained on COCO')

    def one_img2cropfaces(self, img_path):

        det_result = inference_detector(self.detect_model, img_path)[self.det_label]
        det_check = det_result[:, 4] >= self.args.det_score_thr
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR) if isinstance(img_path, str) else img_path
        crop_face_imgs = list()
        if any(det_check):
            det_result = det_result[det_check]
            det_result = [dict(bbox=x) for x in list(det_result)]
            pose = inference_top_down_pose_model(self.pose_model, img_path, det_result, format='xyxy')[0]
            for pos in pose:
                img = facecrop(pos, img_raw, (self.img_size, self.img_size))
                crop_face_imgs.append(img)
        else:
            crop_face_imgs, img_raw, det_check, pose = list(), list(), list(), list()
        return crop_face_imgs, img_raw, det_check, pose

    def cropface2feature(self, images):
        one_feature, prediction_list = dict(), list()
        images = cv2.resize(images, (224, 224))
        images = self._transform(images)
        images = images.to(self.args.device)
        images = torch.stack([images], 0)

        for i, model in enumerate(self.models):
            with torch.no_grad():
                outputs = model(images).cpu()
                outputs = F.softmax(outputs, 1)
                outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                outputs = [round(o, 4) for o in outputs.numpy()]
                prediction_list.append(outputs)
                one_feature[self.model_name[i]] = outputs
        return one_feature, np.array(prediction_list)

    def feature2result(self, feature):
        model_dict, args, model_dict_proba = self.model_dict, self.args, self.args.proba_conf
        test_results_list, tmp_test_result_list = list(), list()

        for idx, (model_name, _) in enumerate(model_dict):
            tmp_test_result_list.append(model_dict_proba[idx] * np.array(feature[idx]))
        tmp_test_result_list = np.array(tmp_test_result_list)
        y_score = np.sum(tmp_test_result_list, axis=0)
        y_pred = np.argmax(y_score, axis=0)

        return y_pred, max(y_score / sum(y_score))


def load_det_model(args):
    detect_model = init_detector(args.det_config, args.det_checkpoint, args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    return detect_model, pose_model


def load_cls_model(path, device, out_class=4):
    checkpoints = glob(os.path.join(path, '*'))
    model_names, model_set = list(), list()
    model_dict = [(i.split('/')[-1].split('__')[0], i.split('/')[-1]) for i in checkpoints if
                  not i.split('/')[-1].split('__')[0] == i.split('/')[-1]]
    # load_label + img_data
    labels, model_set = list(), list()

    for model_name, checkpoint_path in model_dict:
        # each item is 7-ele array
        print("Processing", checkpoint_path)

        model = getattr(models, model_name)
        model_names.append(model_name)
        model = model(in_channels=3, num_classes=out_class)
        state = torch.load(os.path.join(path, checkpoint_path),
                           map_location=lambda storage, loc: storage)
        model.load_state_dict(state["net"])
        model.to(device)
        model.eval()
        model_set.append(model)

    return model_dict, model_names, model_set


def facecrop(p, img_raw, img_size):
    (rx, ry), (lx, ly) = p['keypoints'][0:2, 0:2]
    center = ((lx + rx) // 2, (ly + ry) // 2)
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    scale = EYE_FACE_RATIO / (np.sqrt(((rx - lx) ** 2) + ((ry - ly) ** 2)))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += (img_size[0] / 2 - center[0])
    M[1, 2] += (img_size[1] / 2 + int(img_size[1] / 20) - center[1])
    img = cv2.warpAffine(img_raw, M, img_size, borderValue=0.0)
    return img


def show_result_img(img):
    img = cv2.imencode('.png', img)[1].tostring()
    f = io.BytesIO()
    f.write(img)
    f.seek(0)
    return f


def make_byte_image_2_cv2(image_file):
    image_bytes = image_file.read()
    image_cv = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(image_cv, cv2.IMREAD_COLOR)
    return img


def draw_result(img, pose, label, cat_path, pose_model=None):
    diction = ['행복/즐거움', '중립/안정', '슬픔/두려움', '화남/싫음'] if not cat_path == 'cat' else ['행복/즐거움', '중립/안정', '화남/싫음']
    if pose_model is not None:
        img = vis_pose_result(pose_model, img, [pose])
    bbox = pose['bbox']
    boxx, boxy, box_length = int(bbox[0]), int(bbox[1]), int(bbox[2] + bbox[3] - bbox[0] - bbox[1])
    predict = diction[label]
    img = cv2_draw_korea(img, str(predict), box_length, (boxx, boxy + 12))
    return img


def cv2_draw_korea(cv_img, text, box_length, position=(10, 0), color=(178, 132, 190, 0)):
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    b, g, r, a = color
    org = position
    font = ImageFont.truetype(font=osp.join("tools", "gongso.ttf"), size=int(box_length / 15))  # korea font
    im_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(im_pil)
    draw.text(org, text, (r, g, b, a), font=font)  # because of RGB2BGR
    img = np.array(im_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def for_same_name(path: str, name: str, data_type='jpg') -> str:
    save_path, uni = osp.join(f'{path}', f'{name}.{data_type}'), 1
    while osp.exists(save_path):
        save_path = osp.join(f'{path}', f'{name}_({uni}).{data_type}')
        uni += 1
    return save_path


def who_r_u(score: List[List[int]], result: List[List[dict]], threshold=38.5) -> List[List[dict]]:
    for z, score_cut in enumerate(score):
        for y, scc in enumerate(score_cut):
            if scc < threshold:
                result[z][y]['label'] = 'who are you?'
    return result
