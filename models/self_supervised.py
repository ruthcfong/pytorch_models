import torch
import torch.nn as nn

from torchvision.models.utils import load_state_dict_from_url

from . import *


model_urls = {
    AUDIO: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_audio-eb78354d.pt',  # TODO(ruthfong): Debug backward.
    DEEPCONTEXT: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_deepcontext-983efb8d.pt',
    EGOMOTION: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_egomotion-a0a681d1.pt',  # TODO(ruthfong): Debug backward.
    OBJECTCENTRIC: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_objectcentric-95f20e7e.pt',
    MOVING: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_learningbymoving-6326bbff.pt',
    PUZZLE: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_solvingpuzzle-51f91a86.pt',
    VIDEOORDER: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_videoorder-883e5ce1.pt',
    VIDEOTRACKING: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_videotracking-0ba5696f31.pt',
}


class Power(nn.Module):
    def __init__(self, power=1., scale=1., shift=0.):
        super(Power, self).__init__()
        self.power = power
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return (self.shift + self.scale * x) ** self.power

    def __str__(self):
        return 'Power(power={},scale={},shift={})'.format(self.power,
                                                         self.scale,
                                                         self.shift)


class AlexNet(nn.Module):
    def __init__(self, features=None, classifier=None, task_name=None):
        super(AlexNet, self).__init__()
        self.task_name = task_name
        if task_name is not None:
            assert features is None
            assert classifier is None
            self.set_architecture()
        else:
            assert task_name is None
            self.features = features
            self.classifier = classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        if self.classifier is None:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def set_architecture(self):
        assert self.task_name is not None
        self.classifier = None
        if self.task_name == AUDIO:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                nn.BatchNorm2d(96, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2), groups=2),
                nn.BatchNorm2d(256, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.BatchNorm2d(384, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.BatchNorm2d(384, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.BatchNorm2d(256, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
            ])
        elif self.task_name == DEEPCONTEXT:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4),
                          padding=(5, 5)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
            ])
        elif self.task_name == EGOMOTION:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.BatchNorm2d(96, eps=9.99999974738e-06,
                               momentum=0.999000012875, affine=False,
                               track_running_stats=True),
                nn.LocalResponseNorm(5, alpha=9.99999974738e-05, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=9.99999974738e-05, beta=0.75,
                                     k=1.0),
                nn.BatchNorm2d(256, eps=9.99999974738e-06,
                               momentum=0.999000012875, affine=False,
                               track_running_stats=True),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(384, eps=9.99999974738e-06,
                               momentum=0.999000012875, affine=False,
                               track_running_stats=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(384, eps=9.99999974738e-06,
                               momentum=0.999000012875, affine=False,
                               track_running_stats=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.BatchNorm2d(256, eps=9.99999974738e-06,
                               momentum=0.999000012875, affine=False,
                               track_running_stats=True),
                Power(power=1.0, scale=0.5, shift=0.0),
            ])
            self.classifier = nn.Sequential(*[
                nn.Linear(in_features=9216, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(4096, eps=9.99999974738e-06,
                               momentum=0.999000012875, affine=False,
                               track_running_stats=True),
            ])
        elif self.task_name == MOVING:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=9.99999974738e-05, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=9.99999974738e-05, beta=0.75,
                                     k=1.0),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
            ])
        elif self.task_name == OBJECTCENTRIC:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
            ])
            self.classifier = nn.Sequential(*[
                nn.Linear(in_features=9216, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=1024, bias=True),
                nn.ReLU(inplace=True),
            ])
        elif self.task_name == PUZZLE:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
            ])
        elif self.task_name == VIDEOORDER:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-04, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75,
                                     k=1.0),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.BatchNorm2d(384, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.BatchNorm2d(384, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.BatchNorm2d(256, eps=1e-5,
                               momentum=0.999, affine=False,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
            ])
        elif self.task_name == VIDEOTRACKING:
            self.features = nn.Sequential(*[
                nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-04, beta=0.75,
                                     k=1.0),
                nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1),
                          padding=(2, 2), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
                nn.LocalResponseNorm(5, alpha=1e-04, beta=0.75,
                                     k=1.0),
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                             ceil_mode=True),
            ])
        else:
            raise NotImplementedError


def alexnet(task_name,
            pretrained=False,
            progress=True,
            **kwargs):
    assert task_name in TASK_NAMES
    model = AlexNet(task_name=task_name, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[task_name],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
