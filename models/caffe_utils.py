import os

import caffe
import numpy as np


def set_gpu(gpu=None):
    """Sets GPU device if provided."""
    # TODO(ruthfong): Figure out right assert.
    # assert isinstance(gpu, int)
    if gpu:
        caffe.set_device(gpu)


def get_imagenet_mean():
    """Returns 1D array of length 3 with the mean BGR value for ImageNet."""
    # TODO(ruthfong): Figure out the right path.
    # From here: https://github.com/CSAILVision/NetDissect/blob/release1/script/rundissect.sh#L165
    return np.array([109.5388, 118.6897, 124.6901])
    if False:
        mu = np.load('./data/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)
        if verbose:
            print('mean-subtracted values: ' + zip('BGR', mu))
        return mu


def get_imagenet_transformer(net):
    """Returns :class:`caffe.io.Transformer` for ImageNet preprocessing."""
    # TODO(ruthfong): Compare to https://github.com/CSAILVision/NetDissect/blob/release1/src/loadseg.py#L623-L638
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    mu = get_imagenet_mean()
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    return transformer


def get_caffe_model(prototxt_path,
                    caffemodel_path=None,
                    mode=caffe.TEST):
    """Returns a caffe network.

    Args:
        prototxt_path (str): path to .prototxt file.
        caffemodel_path (str, optional): path to .caffemodel file.
            Default: ``None``.
        mode (str, optional): caffe mode to denote training or inference mode.
            Default: ``caffe.TEST``.

    Returns:
        :class:`caffe.Net`: caffe network.
    """
    assert os.path.exists(prototxt_path)
    if os.path.exists(caffemodel_path):
        net = caffe.Net(prototxt_path, caffemodel_path, mode)
    else:
        net = caffe.Net(prototxt_path, mode)
    return net


def net_forward(net, img_paths, transformer=None):
    """Do a forward pass through a caffe network.

    Args:
        net (:class:`caffe.Net`): caffe network.
        img_paths (str or list of str): image path(s) for input image(s).
        transformer (:class:`caffe.io.Transformer`, optional): caffe
            transformer to handle data preprocessing.

    Returns:
        tuple: tuple containing:
          - :caffe:`caffe.Net`: caffe network after forward pass
          - dict: results from forward pass
    """
    if not transformer:
        transformer = get_imagenet_transformer(net)
    if isinstance(img_paths, str):
        num_imgs = 1
    else:
        num_imgs = len(img_paths)

    net.blobs['data'].reshape(1,
                              3,
                              net.blobs['data'].data.shape[2],
                              net.blobs['data'].data.shape[3])

    if num_imgs == 1:
        img = caffe.io.load_image(img_paths)
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
    else:
        for i in range(num_imgs):
            img = caffe.io.load_images(img_paths[i])
            net.blobs['data'].data[i, ...] = transformer.preprocess('data',
                                                                    img)

    res = net.forward()

    return net, res


def net_backward(net, end_blob, gradient):
    net.blobs[end_blob].diff[...] = gradient
    res = net.backward()

    return net, res

