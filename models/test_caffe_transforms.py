import os

import numpy as np

from .caffe_transforms import CAFFE_IMAGENET_BGR_MEAN
from . import ROOT_DIR


def test_imagenet_mean():
    # Load the mean ImageNet image (as distributed with Caffe) for subtraction.
    mu = np.load(os.path.join(ROOT_DIR, 'data/ilsvrc_2012_mean.npy'))

    # Average over pixels to obtain the mean (BGR) pixel values.
    mu = mu.mean(1).mean(1)

    assert np.abs(CAFFE_IMAGENET_BGR_MEAN - mu).max() < 1e-8
