import os
import numpy as np

# Get root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(
    os.path.realpath(__file__), os.pardir), os.pardir))

DEFAULT_IMG_PATH = os.path.join(ROOT_DIR, 'data/dog_cat.jpeg')


AUDIO = 'weakly_audio'
COLORIZATION = 'weakly_colorization'  # TODO(ruthfong): Add.
DEEPCONTEXT = 'weakly_deepcontext'
EGOMOTION = 'weakly_egomotion'
MOVING = 'weakly_learningbymoving'
OBJECTCENTRIC = 'weakly_objectcentric'
PUZZLE = 'weakly_solvingpuzzle'
SPLITBRAIN = 'weakly_splitbrain'  # TODO(ruthfong): Add.
VIDEOORDER = 'weakly_videoorder'
VIDEOTRACKING = 'weakly_videotracking'


TASK_NAMES = [
    AUDIO,
    COLORIZATION,
    DEEPCONTEXT,
    EGOMOTION,
    MOVING,
    OBJECTCENTRIC,
    PUZZLE,
    SPLITBRAIN,
    VIDEOORDER,
    VIDEOTRACKING,
]


# Mean ImageNet BGR values averaged over spatial locations, as stored in
# https://github.com/BVLC/caffe/blob/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
CAFFE_IMAGENET_BGR_MEAN = np.array([104.00698793, 116.66876762, 122.67891434])
