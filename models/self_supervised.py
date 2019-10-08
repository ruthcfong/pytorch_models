import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


PUZZLE = 'weakly_solvingpuzzle'
model_urls = {
    PUZZLE: 'https://github.com/ruthcfong/pytorch_models/releases/download/v2.0/alexnet-weakly_solvingpuzzle-7e75c18e.pt',
}


class AlexNetPuzzle(nn.Module):
    def __init__(self, params=None):
        super(AlexNetPuzzle, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.Conv2d(96, 256, kernel_size=5, groups=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, groups=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, groups=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        if params:
            self.load_caffe_weights(params)

    def forward(self, x):
        return self.features(x)

    def load_caffe_weights(self, params):
        self.features[0].weight.data[...] = torch.from_numpy(params['conv1_s1'][0].data)
        self.features[0].bias.data[...] = torch.from_numpy(params['conv1_s1'][1].data)
        self.features[4].weight.data[...] = torch.from_numpy(params['conv2_s1'][0].data)
        self.features[4].bias.data[...] = torch.from_numpy(params['conv2_s1'][1].data)
        self.features[8].weight.data[...] = torch.from_numpy(params['conv3_s1'][0].data)
        self.features[8].bias.data[...] = torch.from_numpy(params['conv3_s1'][1].data)
        self.features[10].weight.data[...] = torch.from_numpy(params['conv4_s1'][0].data)
        self.features[10].bias.data[...] = torch.from_numpy(params['conv4_s1'][1].data)
        self.features[12].weight.data[...] = torch.from_numpy(params['conv5_s1'][0].data)
        self.features[12].bias.data[...] = torch.from_numpy(params['conv5_s1'][1].data)


def alexnet(task_name=PUZZLE,
            pretrained=False,
            progress=True,
            **kwargs):
    assert task_name is PUZZLE
    if task_name is PUZZLE:
        model = AlexNetPuzzle(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[task_name],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model