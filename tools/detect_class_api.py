import json
import os.path as osp
import sys
import time

import cv2
import numpy as np
from flask import request, send_file

from .detect_class_tools import make_byte_image_2_cv2, show_result_img, draw_result


def set_class(args, class_name):
    args.test = 'test' if 'test' in class_name else 0
    args.crop_face = True if 'crop' in class_name else False
    class_name = class_name.replace('test', '').replace('crop', '')
    args.dog_cat = 'dog' if '개' in class_name or '강' in class_name else 'cat'
    return args, class_name


def emotion_predict(args, models):
    global show_well_test, img_file
    if not request.method == "POST":
        sys.exit()
    if request.files.getlist("image"):  # and len(request.files.getlist("image")) > 0:  # multi image
        image_files = request.files.getlist('image')
        if request.values.get('classname'):
            class_name = request.values.get('classname')
            args, class_name = set_class(args, class_name)
        result, predicts, confs, pose = dict(), list(), list(), list()
        models.choice_dog_cat(args.dog_cat)
        for file in image_files:
            name = file.filename.split('.')[0]
            init_time = time.time()
            img = make_byte_image_2_cv2(file)
            if not args.crop_face:
                cropfaces, img, box_chk, pose = models.one_img2cropfaces(img)
            else:
                cropfaces = [img]
            show_well_test = args.test == 'test' and len(pose)
            for i, crop_face in enumerate(cropfaces):
                feature, one_feature = models.cropface2feature(crop_face)
                y_prediction, conf = models.feature2result(one_feature)
                predicts.append(y_prediction), confs.append(round(conf, 3))
                if show_well_test:
                    img = draw_result(img, pose[i], y_prediction, args.cat_path, models.pose_model)
                    img_file = show_result_img(img)
                if not args.save_img_path == '':
                    save_result = crop_face if not args.test == 'test' else img
                    cv2.imwrite(osp.join(args.save_img_path, f'{name}_{i}.jpg'), save_result)
            print('processing times = ', time.time() - init_time)

        result["y_pred"], result["confidence"] = predicts, confs
        print(result)
        return json.dumps(result, cls=NpEncoder) if not show_well_test else send_file(img_file, mimetype='image/png')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
