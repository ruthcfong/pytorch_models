import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


PUZZLE = 'weakly_solvingpuzzle'
model_urls = {
    PUZZLE: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_solvingpuzzle-7e75c18e.pt',
}


class AlexNet(nn.Module):
    def __init__(self, features=None, task_name=None):
        super(AlexNet, self).__init__()
        self.task_name = task_name
        if features is not None and task_name is None:
            self.features = features
        elif task_name is not None and features is None:
            self.set_features()
        else:
            assert False

    def set_features(self):
        assert self.task_name is not None
        if self.task_name == PUZZLE:
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
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.features(x)


def alexnet(task_name=PUZZLE,
            pretrained=False,
            progress=True,
            **kwargs):
    if task_name is PUZZLE:
        model = AlexNet(features=None, task_name=task_name, **kwargs)
    else:
        raise NotImplementedError
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[task_name],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model