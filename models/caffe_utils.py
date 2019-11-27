import os

# Only show warnings + errors.
os.environ["GLOG_minloglevel"] = "2"

import caffe

from .caffe_transforms import get_imagenet_transformer


def set_gpu(gpu=None):
    """Sets GPU device if provided."""
    # TODO(ruthfong): Figure out right assert.
    # assert isinstance(gpu, int)
    if gpu is not None:
        caffe.set_device(gpu)


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


def net_forward(net, img_paths, mean_center=True, scale=True, transformer=None):
    """Do a forward pass through a caffe network.

    Args:
        net (:class:`caffe.Net`): caffe network.
        img_paths (str or list of str): image path(s) for input image(s).
        mean_center (bool, optional): If True, mean center image data.
            Default: ``True``.
        transformer (:class:`caffe.io.Transformer`, optional): caffe
            transformer to handle data preprocessing. Default: ``None``.

    Returns:
        tuple: tuple containing:
          - :caffe:`caffe.Net`: caffe network after forward pass
          - dict: results from forward pass
    """
    if not transformer:
        transformer = get_imagenet_transformer(net,
                                               mean_center=mean_center,
                                               scale=scale)
    if isinstance(img_paths, str):
        num_imgs = 1
    else:
        num_imgs = len(img_paths)

    net.blobs['data'].reshape(num_imgs,
                              net.blobs['data'].data.shape[1],
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
    """Do a backward pass through a caffe network."""
    net.blobs[end_blob].diff[...] = gradient
    res = net.backward()

    return net, res

