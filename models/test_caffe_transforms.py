import os

import numpy as np

from .caffe_transforms import CAFFE_IMAGENET_BGR_MEAN


def test_imagenet_mean():
    # Get root directory.
    root_dir = os.path.abspath(os.path.join(os.path.join(
        os.path.realpath(__file__), os.pardir), os.pardir))

    # Load the mean ImageNet image (as distributed with Caffe) for subtraction.
    mu = np.load(os.path.join(root_dir, 'data/ilsvrc_2012_mean.npy'))
    # Average over pixels to obtain the mean (BGR) pixel values.
    mu = mu.mean(1).mean(1)

    assert np.abs(CAFFE_IMAGENET_BGR_MEAN - mu).max() < 1e-8
