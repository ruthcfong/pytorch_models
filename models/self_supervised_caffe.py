import argparse
import os

import caffe_utils as cutils

from .self_supervised import PUZZLE

DEFAULT_CAFFE_MODELS_DIR = '/scratch/local/ssd/ruthfong/models/net_dissect/caffe'
DEFAULT_IMG_PATH = '../dog_cat.jpeg'


def convert_caffe_to_pytorch(img_path=DEFAULT_IMG_PATH,
                             task_name=PUZZLE,
                             caffe_models_dir=DEFAULT_CAFFE_MODELS_DIR,
                             gpu=None):
    cutils.set_gpu(gpu)
    prototxt_path = os.path.join(caffe_models_dir,
                                 task_name + '.prototxt')
    caffemodel_path = os.path.join(caffe_models_dir,
                                   task_name + '.caffemodel')
    net = cutils.get_caffe_model(prototxt_path,
                                 caffemodel_path=caffemodel_path)
    net = cutils.net_forward(net, img_path)
    import pdb; pdb.set_trace();
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name',
                        type=str,
                        default=PUZZLE)
    parser.add_argument('--caffe_models_dir',
                        type=str,
                        default=DEFAULT_CAFFE_MODELS_DIR)
    parser.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()
    convert_caffe_to_pytorch(task_name=args.task_name,
                             caffe_models_dir=args.caffe_models_dir,
                             gpu=args.gpu)
