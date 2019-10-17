# pytorch_models

## Self-supervised Models
In [models/self_supervised.py](models/self_supervised.py), we provide a stand-alone file that provides model definitions for self-supervised models converted from Caffe to PyTorch.

`model = models.self_supervised.alexnet(task_name='weakly_solvingpuzzle', pretrained=True)`

The following models have been converted and fully verified to be equivalent:
* `weakly_deepcontext`
* `weakly_learningbymoving`
* `weakly_objectcentric`
* `weakly_solvingpuzzle`
* `weakly_videoorder`
* `weakly_videotracking`

These models have been verified to be equivalent in the forward pass but have discrepancies in the backwards pass (i.e., gradients):
* `weakly_audio`
* `weakly_egomotion`

TODO-models:
* `weakly_colorization`
* `weakly_splitbrain`

TODO-misc:
* define `get_transform` to provide the right data transformations for each task.

Models were taken from [http://netdissect.csail.mit.edu](http://netdissect.csail.mit.edu).
