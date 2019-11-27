import pytest
import numpy as np

from . import *   # Import task names and default image path.

from .transforms import get_transform
from .caffe_transforms import get_transformer, TRANSFORMERS
from .self_supervised_caffe import get_caffe_model

from PIL import Image


@pytest.mark.parametrize("task_name", TRANSFORMERS.keys())
def test_caffe_pytorch_equivalency(task_name):
    caffe_net = get_caffe_model(task_name)
    caffe_transformer = get_transformer(task_name, caffe_net)

    caffe_net.blobs["data"].reshape(1, 3, 227, 227)
    x_caffe = caffe_transformer.preprocess("data", DEFAULT_IMG_PATH)

    # torch_transform = get_transform(task_name=task_name, size=227)
    torch_transform = get_transform(task_name=task_name, size=227)
    x_torch = torch_transform(Image.open(DEFAULT_IMG_PATH).convert("RGB"))

    # Switch channels to be BGR.
    x_torch = x_torch[[2,1,0]]

    # Check relative difference.
    max_diff = np.abs(x_caffe - x_torch.cpu().data.numpy()).max()
    mag = x_caffe.max()
    rel_diff = max_diff / mag

    assert rel_diff < 2e-6
