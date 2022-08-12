import argparse
import os
from os import path as osp

from flask import Flask, render_template
from tools import detect_class_api
from tools.detect_class_tools import DogFaceDetect

app = Flask(__name__)
Recognition_URL = osp.join("/", "dog", "recognition")


@app.route('/dog-rec/', methods=["GET", "POST"])
# @app.route('/', methods=["GET","POST"])
def index1():
    return render_template('emotion.html')


@app.route(Recognition_URL, methods=["POST"])
def service1():
    return detect_class_api.emotion_predict(args, Models)


def parse_args():
    import ast
    def arg_as_list(s):
        v = ast.literal_eval(s)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % s)
        return v

    parser = argparse.ArgumentParser(description='dog-emotionset')
    parser.add_argument("--port", default=3334, type=int, help="port number")
    # det
    parser.add_argument('--det-config', default='detect/yolox_x_8x8_300e_coco.py',
                        help='Dog detection config file path (from mmdet)')
    parser.add_argument('--det-checkpoint',
                        default='detect/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
                        help='dog detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='detect/hrnet_w32_animalpose_256x256.py',
        help='dog pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='detect/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth'
        , help='dog pose estimation checkpoint file/url')
    parser.add_argument('--det-score-thr', type=float, default=0.2, help='the threshold of dog detection score')
    parser.add_argument('--device', type=str, default='cuda:1', help='CPU/CUDA device option')
    parser.add_argument('--dog_cat', default='dog', help='dog or cat')
    parser.add_argument('--dog_path', default='real', help='dog or cat')
    parser.add_argument('--cat_path', default='cat', help='dog or cat')
    parser.add_argument('--output_size', type=int, default=256, help='img_size')
    # recog
    parser.add_argument("--save_img_path", default="./saved/test", type=str, help="if you wanna save image, put path")
    parser.add_argument("--test", default='not_test', help="if show one img, write 'test' ")
    parser.add_argument('--crop_face', action='store_true', help='if input is crop dog face')
    parser.add_argument('--cls_check_path', default='checkpoint', help='input checkpoint path')
    parser.add_argument('--config', default='classification/configs/dog_config.json',
                        help='input config path')
    parser.add_argument('--proba_conf', default=[1.2, 1.4, 0.3], type=arg_as_list, help='input config path')

    # cls_check_path/dog_path == "checkpoint/real", cls_check_path/cat_path == "checkpoint/cat"
    # proba_conf : dog = [1.2, 1.4, 0.3], cat = [1.4, 0.1, 1.4]
    # check! this list need to align... not perfect list -> all of ensemble models need to change this list
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Models = DogFaceDetect(args)
    os.makedirs(args.save_img_path, exist_ok=True)
    app.run(host="0.0.0.0", port=args.port)
