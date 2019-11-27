from PIL import Image
from scipy.ndimage import zoom
from skimage import color
from skimage.transform import resize
import torch
from torchvision import transforms

# import task names and default caffe bgr mean.
from . import  *


BGR_MEANS = {
    AUDIO: [104., 117., 123.],
}


class CaffeColorization(object):
    def __init__(self, mean=50):
        self.mean = mean

    def __call__(self, x):
        x  = color.rgb2lab(x)  # lab image
        x = x[:,:,0]  # l image: [0, 100]
        x = x - self.mean  # mean-center
        return x


class CaffeChannelSwap(object):
    def __init__(self, order=(2, 1, 0)):
        self.order = order

    def __call__(self, x):
        assert len(x.shape) == 3
        assert x.shape[0] == 3
        return x[self.order, :, :]


class CaffeChannelNormalize(object):
    """Normalize a tensor per-color channel (used for deep context)."""
    def __call__(self, x):
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 3
        assert x.shape[0] == 3
        mu = x.mean(dim=(1,2))
        x = x - mu.unsqueeze(1).unsqueeze(2)
        return x


class CaffeDeepContextNormalize(object):
    """
    Normalize the mean and variance so that gradients are a less useful cue;
    then scale (by 50) so that the variance is roughly the same as the usual
    AlexNet inputs.
    """
    def __init__(self, scale=50.):
        self.scale = scale

    def __call__(self, x):
        assert isinstance(x, torch.Tensor)
        z = x.pow(2).mean().sqrt()
        return x / z * self.scale


class CaffeScale(object):
    def __init__(self, scale=1./255.):
        self.scale = scale
        pass

    def __call__(self, x):
        assert isinstance(x, Image.Image)
        return np.array(x) * self.scale


class CaffeResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        try:
            iter(size)
        except TypeError as te:
            print(str(size) + 'is not iterable')
        self.size = size

    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        return self.resize_image(x, self.size)


    @staticmethod
    def resize_image(im, new_dims, interp_order=1):
        """
        Resize an image array with interpolation.
        Parameters
        ----------
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        interp_order : interpolation order, default is linear.
        Returns
        -------
        im : resized ndarray with shape (new_dims[0], new_dims[1], K)
        """

        if im.shape[-1] == 1 or im.shape[-1] == 3:
            im_min, im_max = im.min(), im.max()
            if im_max > im_min:
                # skimage is fast but only understands {1,3} channel images
                # in [0, 1].
                im_std = (im - im_min) / (im_max - im_min)
                resized_std = resize(im_std, new_dims, order=interp_order,
                                     mode='constant')
                resized_im = resized_std * (im_max - im_min) + im_min
            else:
                # the image is a constant -- avoid divide by 0
                ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                               dtype=np.float32)
                ret.fill(im_min)
                return ret
        else:
            # ndimage interpolates anything but more slowly.
            scale = tuple(np.array(new_dims, dtype=float)
                          / np.array(im.shape[:2]))
            resized_im = zoom(im, scale + (1,), order=interp_order)
        return resized_im.astype(np.float32)



def get_transform(task_name, size=None):
    """Get pytorch transform for ported caffe model (copied from TorchRay)."""
    # TODO(ruthfong): Compare to https://github.com/CSAILVision/NetDissect/blob/release1/src/loadseg.py#L623-L638
    # From here: https://github.com/CSAILVision/NetDissect/blob/release1/script/rundissect.sh#L165

    if task_name == COLORIZATION:
        transform = transforms.Compose([
            CaffeResize(size),
            CaffeColorization(),
            transforms.ToTensor(),
        ])
        return transform
    elif task_name == DEEPCONTEXT:
        transform = transforms.Compose([
            CaffeScale(scale=1.),
            CaffeResize(size),
            transforms.ToTensor(),
            CaffeChannelNormalize(),
            CaffeDeepContextNormalize(),
            CaffeChannelSwap(), # TODO(ruthfong): Change saved model to remove this.
        ])
        return transform

    if task_name in BGR_MEANS:
        bgr_mean = BGR_MEANS[task_name]
    else:
        bgr_mean = CAFFE_IMAGENET_BGR_MEAN

    # TODO(ruthfong): Remove this.
    # bgr_mean = [109.5388, 118.6897, 124.6901]  # net dissect
    # bgr_mean = [104.00698793, 116.66876762, 122.67891434] # caffe
    # bgr_mean = [103.939, 116.779, 123.68]  # torchray

    mean = [m / 255. for m in reversed(bgr_mean)]
    std = [1 / 255.] * 3

    transform = transforms.Compose([
        CaffeScale(scale=1/255.),
        CaffeResize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform
