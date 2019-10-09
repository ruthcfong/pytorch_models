import argparse
import os

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import torch
import torch.nn as nn

import caffe_utils as cutils

from .self_supervised import PUZZLE, AlexNet

DEFAULT_CAFFE_MODELS_DIR = '/scratch/local/ssd/ruthfong/models/net_dissect/caffe'
DEFAULT_PYTORCH_MODELS_DIR = '/scratch/local/ssd/ruthfong/models/net_dissect/pytorch'
DEFAULT_IMG_PATH = '../dog_cat.jpeg'

DEFAULT_MAX_DIFF_THRESHOLD = 1e-4

# default caffe values
POOL_MAX = 0
POOL_AVE = 1
POOL_STOCHASTIC = 2


def extract_param(param, default_value=None):
    if isinstance(param, int) or isinstance(param, float):
        return param
    if len(param) == 1:
        return int(param[0])
    elif len(param) == 0:
        if default_value is not None:
            return default_value
        else:
            # TODO(ruthfong): raise specific error.
            assert False
    else:
        raise NotImplementedError


get_stride = lambda p: extract_param(p.stride, 1)
get_padding = lambda p: extract_param(p.pad, 0)
get_kernel_size = lambda p: extract_param(p.kernel_size)
get_bias = lambda p: extract_param(p.bias_term, True)
get_groups = lambda p: extract_param(p.group, 1)
get_local_size = lambda p: extract_param(p.local_size, 5)
get_alpha = lambda p: extract_param(p.alpha, 1.)
get_beta = lambda p: extract_param(p.beta, 0.75)
get_k = lambda p: extract_param(p.k, 1.)


def get_net_proto(prototxt_path):
    net_p = caffe_pb2.NetParameter()

    with open(prototxt_path) as f:
        s = f.read()
        txtf.Merge(s, net_p)

    return net_p


def get_torch_net(net_p):
    prev_dim = 3
    inplace = True
    new_layers = []
    learnable_layers = {}

    for i, layer in enumerate(net_p.layer):
        if layer.type == 'Convolution':
            p = layer.convolution_param
            kernel_size = get_kernel_size(p)
            stride = get_stride(p)
            padding = get_padding(p)
            bias = get_bias(p)
            groups = get_groups(p)
            new_layer = nn.Conv2d(prev_dim,
                                  p.num_output,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=1,
                                  groups=groups,
                                  bias=bias,
                                  padding_mode='zeros')
            assert len(layer.top) == 1
            learnable_layers[layer.top[0]] = i
            prev_dim = p.num_output
        elif layer.type == 'ReLU':
            assert len(layer.top) == 1
            assert len(layer.bottom) == 1
            new_layer = nn.ReLU(inplace=inplace if inplace else layer.top[0] == layer.bottom[0])
        elif layer.type == 'Pooling':
            p = layer.pooling_param
            kernel_size = get_kernel_size(p)
            stride = get_stride(p)
            padding = get_padding(p)
            ceil_mode = True  # caffe uses ceil_mode=True
            if p.pool is POOL_MAX:
                new_layer = nn.MaxPool2d(kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=1,
                                         return_indices=False,
                                         ceil_mode=ceil_mode)
            else:
                raise NotImplementedError
        elif layer.type == 'LRN':
            p = layer.lrn_param
            local_size = get_local_size(p)
            alpha = get_alpha(p)
            beta = get_beta(p)
            k = get_k(p)
            new_layer = nn.LocalResponseNorm(local_size,
                                             alpha=alpha,
                                             beta=beta,
                                             k=k)
        else:
            raise NotImplementedError
        print(i, layer)#, new_layer)
        new_layers.append(new_layer)

        features = nn.Sequential(*new_layers)

    return features, learnable_layers


def print_sequential(sequential):
    assert isinstance(sequential, nn.Sequential)
    str_rep = 'Sequential(*[\n'
    for m in sequential:
        prefix = 'nn.' if 'torch.nn' in repr(type(m)) else ''
        str_rep += prefix + str(m) + ',\n'
    str_rep += '])'
    print(str_rep)


def load_weights(caffe_net, torch_net, learnable_layers):
    for param_name, param in caffe_net.params.items():
        layer_i = learnable_layers[param_name]
        num_params = len(param)
        assert num_params > 0
        assert num_params < 3
        assert 'conv' in param_name
        torch_net[layer_i].weight.data[...] = torch.from_numpy(param[0].data)
        if num_params == 2:
            torch_net[layer_i].bias.data[...] = torch.from_numpy(param[1].data)
    return torch_net


def convert_caffe_to_pytorch(img_path=DEFAULT_IMG_PATH,
                             task_name=PUZZLE,
                             caffe_models_dir=DEFAULT_CAFFE_MODELS_DIR,
                             pytorch_models_dir=DEFAULT_PYTORCH_MODELS_DIR,
                             max_diff_threshold=DEFAULT_MAX_DIFF_THRESHOLD,
                             gpu=None):
    # TODO(ruthfong): Check GPU placement.
    cutils.set_gpu(gpu)

    # Get caffe network.
    prototxt_path = os.path.join(caffe_models_dir,
                                 task_name + '.prototxt')
    caffemodel_path = os.path.join(caffe_models_dir,
                                   task_name + '.caffemodel')
    caffe_net = cutils.get_caffe_model(prototxt_path,
                                       caffemodel_path=caffemodel_path)

    # Conver caffe network into pytorch feature extractor.
    net_p = get_net_proto(prototxt_path)
    features, learnable_layers = get_torch_net(net_p)
    features = load_weights(caffe_net, features, learnable_layers)

    # Print features definition.
    print('Copy this into models.self_supervised.AlexNet.set_features')
    print_sequential(features)

    # Construct pytorch AlexNet network.
    torch_net = AlexNet(features=features)

    # Forward pass through caffe network.
    caffe_net, caffe_res = cutils.net_forward(caffe_net, img_path)
    assert len(caffe_res) == 1
    caffe_key = caffe_res.keys()[0]

    # Forward pass through pytorch network.
    x = torch.from_numpy(caffe_net.blobs['data'].data)
    torch_res = torch_net(x)

    # Check maximum difference.
    max_diff = (torch_res.cpu().data.numpy() - torch_res[caffe_key].data).max()
    print('Max Diff (' + caffe_key + '): ' + max_diff)
    assert max_diff < max_diff_threshold

    # Save path.
    save_path = os.path.join(pytorch_models_dir,
                             'alexnet-' + task_name + '.pt')
    torch.save(torch_net.state_dict(), save_path)
    print('Saved pytorch model to ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name',
                        type=str,
                        default=PUZZLE)
    parser.add_argument('--caffe_models_dir',
                        type=str,
                        default=DEFAULT_CAFFE_MODELS_DIR)
    parser.add_argument('--max_diff_threshold',
                        type=float,
                        default=DEFAULT_MAX_DIFF_THRESHOLD)
    parser.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()
    convert_caffe_to_pytorch(task_name=args.task_name,
                             caffe_models_dir=args.caffe_models_dir,
                             max_diff_threshold=args.max_diff_threshold,
                             gpu=args.gpu)
