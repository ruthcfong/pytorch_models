import os

# Only show warnings + errors.
os.environ["GLOG_minloglevel"] = "2"

import caffe
from PIL import Image
import pytest
from torchvision import transforms

from . import *   # Import task names and default image path.
from .transforms import get_transform, CaffeResize, CaffeScale
from .caffe_transforms import get_transformer, TRANSFORMERS
from .self_supervised_caffe import get_caffe_model


def _load_caffe_image():
    return caffe.io.load_image(DEFAULT_IMG_PATH)


def _load_torch_image():
    return Image.open(DEFAULT_IMG_PATH).convert("RGB")


def _check_relative_difference(x, y, tolerance=1e-10):
    max_diff = np.abs(x - y).max()
    mag = min(x.max(), y.max())
    rel_diff = max_diff / mag

    assert rel_diff < tolerance


@pytest.mark.parametrize("task_name", TRANSFORMERS.keys())
def test_caffe_pytorch_equivalency(task_name):
    caffe_net = get_caffe_model(task_name)
    caffe_transformer = get_transformer(task_name, caffe_net)

    caffe_net.blobs["data"].reshape(1, 3, 227, 227)
    x_caffe = caffe_transformer.preprocess("data", DEFAULT_IMG_PATH)

    torch_transform = get_transform(task_name=task_name, size=227)
    x_torch = torch_transform(_load_torch_image())

    # Switch channels to be BGR.
    x_torch = x_torch[[2,1,0]]

    # Check relative difference.
    _check_relative_difference(x_caffe, x_torch.cpu().data.numpy(), 2e-6)


def test_caffe_scale_equivalency():
    x_caffe = _load_caffe_image()

    torch_transform = CaffeScale(scale=1/255.)
    x_torch = torch_transform(_load_torch_image())

    _check_relative_difference(x_caffe, x_torch, 1e-7)


@pytest.mark.parametrize("size", [112, 224, 227])
def test_caffe_resize_equivalency(size):
    if isinstance(size, int):
        size = (size, size)
    try:
        iter(size)
    except:
        assert False

    x_caffe = caffe.io.resize_image(_load_caffe_image(), size)

    torch_transform = transforms.Compose([
        CaffeScale(scale=1/255.),
        CaffeResize(size),
    ])
    x_torch = torch_transform(_load_torch_image())

    _check_relative_difference(x_caffe, x_torch, 1e-7)
