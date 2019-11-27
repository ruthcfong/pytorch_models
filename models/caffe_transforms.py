import caffe
import numpy as np
from scipy import misc

# Import names of self-supervised tasks and default bgr mean.
from . import *


class CaffeTransformer(object):
    """
    Parent class to handle data pre-processing for caffe. Child
    classes can use it in one of two ways:

    1., to handle custom data pre-pre-processing by implementing
    :func:`prep_image` and setting :attr:`transformer` to None,

    2., to wrap around :class:`caffe.io.Transformer` by setting
    :attr:`transformer` during initialization.
    """
    def __init__(self, net, blob="data"):
        self.net = net
        self.transformer = None
        self.blob = blob

    def load_image(self, path):
        assert isinstance(path, str)
        return caffe.io.load_image(path)

    def prep_image(self, x):
        raise NotImplementedError

    def preprocess(self, blob, x):
        """Apply preprocessing and load image if necessary.

        Args:
            blob (str): Name of blob.
            x: Path to image (str) or loaded image from :func:`prep_image`.

        Returns:
            (:class:`numpy.ndarray`): pre-processed image.
        """
        assert blob == self.blob

        # Load image if x is a string.
        if isinstance(x, str):
            x = self.load_image(x)

        # Use caffe.io.Transformer if provided.
        if self.transformer is not None:
            x = self.transformer.preprocess(blob, x)
        # Otherwise, use custom prep_image function.
        else:
            x = self.prep_image(x)

        assert isinstance(x, np.ndarray)

        return x


class AudioCaffeTransformer(CaffeTransformer):
    def __init__(self, *args, **kwargs):
        """Different mean center values from prototxt."""
        super(AudioCaffeTransformer, self).__init__(*args, **kwargs)
        self.bgr_mean = np.array([104., 117., 123.])
        self.transformer = get_imagenet_transformer(self.net,
                                                    blob=self.blob,
                                                    mean_center=True,
                                                    scale=True,
                                                    mu=self.bgr_mean)


class DeepContextCaffeTransformer(CaffeTransformer):
    def load_image(self, path):
        return misc.imread(path).astype(float)

    def prep_image(self, x):
        """
        Modified from source code:
        https://github.com/cdoersch/deepcontext/blob/master/train.py#L89-L111
        """
        # Resize input if necessary.
        input_spatial_shape = x.shape[:2]
        target_spatial_shape = self.net.blobs[self.blob].shape[2:]
        if input_spatial_shape != target_spatial_shape:
            x = caffe.io.resize_image(x, target_spatial_shape)

        for i in range(0, 3):
            x[:, :, i] -= np.mean(x[:, :, i])

        # Normalize the mean and variance so that gradients are a less useful cue;
        # then scale by 50 so that the variance is roughly the same as the usual
        # AlexNet inputs.
        x = x / np.sqrt(np.mean(np.square(x))) * 50
        return x.transpose(2, 0, 1)


class DefaultCaffeTransformer(CaffeTransformer):
    def __init__(self, *args, **kwargs):
        super(DefaultCaffeTransformer, self).__init__(*args, **kwargs)
        self.transformer = get_imagenet_transformer(self.net,
                                                    blob=self.blob,
                                                    mean_center=True,
                                                    scale=True,
                                                    mu=None)


TRANSFORMERS = {
    AUDIO: AudioCaffeTransformer,
    DEEPCONTEXT: DeepContextCaffeTransformer,
    OBJECTCENTRIC: DefaultCaffeTransformer,
}


def get_transformer(task_name, net, blob="data"):
    """Returns :class:`CaffeTransformer` to handle pre-processing.

    Args:
        task_name (str): name of self-supervised task.
        net (:class:`caffe.Net`): caffe network.
        blob (str, optional): name of blob. Default: ``"data"``.
        resize (bool): If True, resize image. Default:: ``False``.
    """
    assert task_name in TASK_NAMES

    if not task_name in TRANSFORMERS:
        return DefaultCaffeTransformer(net=net, blob=blob)
        # TODO(ruthfong): Remove.
        if False:
            raise NotImplementedError("{} is not currently supported; {} is the "
                                      "list of supported tasks.".format(
                task_name,
                str(TRANSFORMERS.keys())
            ))
    else:
        transformer = TRANSFORMERS[task_name](net=net,
                                              blob=blob)
        return transformer


def get_imagenet_mean():
    """Returns 1D array of length 3 with the mean BGR value for ImageNet."""
    return CAFFE_IMAGENET_BGR_MEAN
    # TODO(ruthfong): Delete.
    # TODO(ruthfong): Figure out the right path.
    # From here: https://github.com/CSAILVision/NetDissect/blob/release1/script/rundissect.sh#L165
    # return np.array([109.5388, 118.6897, 124.6901])
    if False:
        mu = np.load('./data/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)
        if verbose:
            print('mean-subtracted values: ' + zip('BGR', mu))
        return mu


def get_imagenet_transformer(net,
                             blob="data",
                             mean_center=True,
                             scale=True,
                             mu=None):
    """Returns :class:`caffe.io.Transformer` for ImageNet preprocessing.

    Args:
        net (:class:`caffe.Net`): caffe network.
        blob (str): name of blob at which to do pre-processing.
        mean_center (bool, optional): If True, mean center the image.
            Default: ``True``.
        scale (bool, optional): If True, set the raw scale to 255.
            Default: ``True``.
        mu (list or :class:`numpy.ndarray`, optional): If provided, use instead
            of the default ImageNet mean. Default: ``None``.
    """
    # TODO(ruthfong): Compare to https://github.com/CSAILVision/NetDissect/blob/release1/src/loadseg.py#L623-L638

    # Set shape.
    transformer = caffe.io.Transformer({blob: net.blobs[blob].data.shape})

    # Prepare to re-order axes to be CHW (channel, height, width).
    transformer.set_transpose(blob, (2,0,1))

    # Prepare to mean center image.
    if mean_center:
        if mu is None:
            mu = get_imagenet_mean()
        transformer.set_mean(blob, mu)

    # Prepare to scale image.
    if scale:
        transformer.set_raw_scale(blob, 255)

    # Prepare to swap color channels to pass in BGR image.
    if net.blobs[blob].data.shape[1] == 3:
        transformer.set_channel_swap(blob, (2,1,0))
    else:
        assert net.blobs[blob].data.shape[1] == 1

    return transformer
