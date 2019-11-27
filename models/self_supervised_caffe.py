import argparse
import os

# Only show warnings + errors.
os.environ["GLOG_minloglevel"] = "2"
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

import caffe_utils as cutils

from .caffe_transforms import get_transformer
from .common import attach_debug_probes
from .self_supervised import COLORIZATION, PUZZLE, TASK_NAMES, AlexNet, alexnet, Power
from .transforms import get_transform

DEFAULT_CAFFE_MODELS_DIR = '/scratch/local/ssd/ruthfong/models/net_dissect/caffe'
DEFAULT_PYTORCH_MODELS_DIR = '/scratch/local/ssd/ruthfong/models/net_dissect/pytorch'
DEFAULT_IMG_PATH = './data/dog_cat.jpeg'

DEFAULT_REL_MAX_ACT_DIFF_THRESHOLD = 1e-6
DEFAULT_REL_MAX_GRAD_DIFF_THRESHOLD = 1e-6

# default caffe values
POOL_MAX = 0
POOL_AVE = 1
POOL_STOCHASTIC = 2

def normalize(x):
    x_max = x.max()
    x_min = x.min()
    return (x - x_min) / (x_max - x_min)


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
get_eps = lambda p: extract_param(p.eps, 1e-5)
get_momentum = lambda p: extract_param(p.moving_average_fraction, 0.999)
get_track_running_stats = lambda p: extract_param(p.use_global_stats, False)
get_power = lambda p: extract_param(p.power, 1)
get_scale = lambda p: extract_param(p.scale, 1)
get_shift = lambda p: extract_param(p.shift, 0)
get_dropout_p = lambda p: extract_param(p.dropout_ratio, 0.5)


def get_net_proto(prototxt_path):
    net_p = caffe_pb2.NetParameter()

    with open(prototxt_path) as f:
        s = f.read()
        txtf.Merge(s, net_p)

    return net_p


def get_torch_net(net_p, num_input_channels=3):
    prev_dim = num_input_channels
    inplace = True
    new_features_layers = []
    learnable_features_layers = {}
    new_classifier_layers = []
    learnable_classifier_layers = {}
    first_conv = True
    first_conv_i = 0
    first_fc = True
    first_fc_i = None

    for layer_i, layer in enumerate(net_p.layer):
        if layer.type == 'InnerProduct' and first_fc:
            prev_dim *= 6 * 6
            first_fc = False
            first_fc_i = layer_i
        if layer.type == 'Convolution' and first_conv:
            first_conv_i = layer_i
            first_conv = False
        new_layers = new_features_layers if first_fc else new_classifier_layers
        learnable_layers = learnable_features_layers if first_fc else learnable_classifier_layers
        if first_fc:
            i = layer_i - first_conv_i
        else:
            i = layer_i - first_fc_i - first_conv_i
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
            learnable_layers[layer.name] = (i, layer.type)
            prev_dim = p.num_output
        elif layer.type == 'InnerProduct':
            p = layer.inner_product_param
            bias = get_bias(p)
            new_layer = nn.Linear(prev_dim,
                                  p.num_output,
                                  bias=bias)
            learnable_layers[layer.name] = (i, layer.type)
            prev_dim = p.num_output
        elif layer.type == 'BatchNorm':
            p = layer.batch_norm_param
            momentum = get_momentum(p)
            eps = get_eps(p)
            track_running_stats = get_track_running_stats(p)
            batch_norm_func = nn.BatchNorm2d if first_fc else nn.BatchNorm1d
            new_layer = batch_norm_func(prev_dim,
                                        eps=eps,
                                        momentum=momentum,
                                        track_running_stats=track_running_stats,
                                        affine=False)
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
        elif layer.type == 'Power':
            p = layer.power_param
            power = get_power(p)
            scale = get_scale(p)
            shift = get_shift(p)
            new_layer = Power(power=power,
                              scale=scale,
                              shift=shift)
        elif layer.type == 'Dropout':
            p = layer.dropout_param
            dropout_p = get_dropout_p(p)
            new_layer = nn.Dropout(p=dropout_p)
        elif layer.type == 'Input':
            continue
        elif (layer.type == 'Python'
              and layer.python_param.layer == 'BGR2LabLayer'):
            import pdb; pdb.set_trace();
        else:
            # TODO(ruthfong): Remove
            print(layer.type)
            import pdb; pdb.set_trace();
            raise NotImplementedError
        print(i, layer)
        new_layers.append(new_layer)
        # TODO(ruthfong): Remove after debugging.
        learnable_layers[layer.name] = (i, layer.type)
        if False:
            if layer.top[0] not in learnable_layers:
                 learnable_layers[layer.top[0]] = (i, layer.type)

    features = nn.Sequential(*new_features_layers)
    if new_classifier_layers:
        classifier = nn.Sequential(*new_classifier_layers)
    else:
        classifier = None

    return features, classifier, learnable_features_layers, learnable_classifier_layers


def print_sequential(sequential):
    assert isinstance(sequential, nn.Sequential)
    str_rep = 'nn.Sequential(*[\n'
    for m in sequential:
        prefix = 'nn.' if 'torch.nn' in repr(type(m)) else ''
        str_rep += prefix + str(m) + ',\n'
    str_rep += '])'
    print(str_rep)


def load_weights(caffe_net, torch_net, learnable_layers):
    for i, (param_name, param) in enumerate(caffe_net.params.items()):
        if param_name not in learnable_layers:
            print('Skipping ' + param_name)
            continue
        layer_i, layer_type = learnable_layers[param_name]
        num_params = len(param)
        assert num_params > 0
        assert num_params < 4
        try:
            assert layer_type in ['Convolution', 'InnerProduct', 'BatchNorm']
        except:
            print(param_name)
            import pdb; pdb.set_trace();
        if layer_type == 'BatchNorm':
            assert num_params == 3
            torch_net[layer_i].running_mean.data[...] = torch.from_numpy(
                param[0].data) / param[2].data[0]
            torch_net[layer_i].running_var.data[...] = torch.from_numpy(
                param[1].data) / param[2].data[0]
        else:
            assert num_params < 3
            if i == 0 and param[0].data.shape[1] == 3:
                torch_net[layer_i].weight.data[...] = torch.from_numpy(
                    param[0].data[:, [2, 1, 0], :, :])
            else:
                torch_net[layer_i].weight.data[...] = torch.from_numpy(
                    param[0].data)
            if num_params == 2:
                torch_net[layer_i].bias.data[...] = torch.from_numpy(
                    param[1].data)
    return torch_net


def get_caffe_model(task_name=PUZZLE,
                    caffe_models_dir=DEFAULT_CAFFE_MODELS_DIR,
                    ):
    prototxt_path = os.path.join(caffe_models_dir,
                                 task_name + '.prototxt')
    caffemodel_path = os.path.join(caffe_models_dir,
                                   task_name + '.caffemodel')
    caffe_net = cutils.get_caffe_model(prototxt_path,
                                       caffemodel_path=caffemodel_path)
    return caffe_net


def convert_caffe_to_pytorch(img_path=DEFAULT_IMG_PATH,
                             task_name=PUZZLE,
                             caffe_models_dir=DEFAULT_CAFFE_MODELS_DIR,
                             pytorch_models_dir=DEFAULT_PYTORCH_MODELS_DIR,
                             max_act_diff_threshold=DEFAULT_REL_MAX_ACT_DIFF_THRESHOLD,
                             max_grad_diff_threshold=DEFAULT_REL_MAX_GRAD_DIFF_THRESHOLD,
                             test_conversion=False,
                             use_caffe_processing=False,
                             plot_grad=False,
                             debug=False,
                             gpu=None):
    # TODO(ruthfong): Check GPU placement.
    cutils.set_gpu(gpu)

    # Get caffe network.
    caffe_net = get_caffe_model(task_name=task_name,
                                caffe_models_dir=caffe_models_dir)

    if test_conversion:
        torch_net = alexnet(task_name=task_name, pretrained=True)
        if debug:
            prototxt_path = os.path.join(caffe_models_dir,
                                         task_name + '.prototxt')
            net_p = get_net_proto(prototxt_path)
            num_input_channels = 1 if task_name == COLORIZATION else 3
            _, _, learnable_features, learnable_classifier = get_torch_net(net_p, num_input_channels=num_input_channels)
    else:
        # Convert caffe network into pytorch feature extractor.
        prototxt_path = os.path.join(caffe_models_dir,
                                     task_name + '.prototxt')
        net_p = get_net_proto(prototxt_path)
        num_input_channels = 1 if task_name == COLORIZATION else 3
        features, classifier, learnable_features, learnable_classifier = get_torch_net(net_p,
                                                                                       num_input_channels=num_input_channels)
        # import pdb; pdb.set_trace();
        print('Loading features weights')
        features = load_weights(caffe_net, features, learnable_features)
        if classifier:
            print('Loading classifier weights')
            classifier = load_weights(caffe_net, classifier, learnable_classifier)

        # Print features definition.
        print('Copy this into models.self_supervised.AlexNet.set_architecture')
        print('self.features = ')
        print_sequential(features)
        if classifier:
            print('self.classifier = ')
            print_sequential(classifier)

        # Construct pytorch AlexNet network.
        torch_net = AlexNet(features=features, classifier=classifier)
    torch_net.eval()

    # Forward pass through caffe network.
    transformer = get_transformer(task_name=task_name, net=caffe_net)
    caffe_net, caffe_res = cutils.net_forward(caffe_net, img_path, transformer=transformer)
    assert len(caffe_res) == 1
    caffe_end_blob = caffe_res.keys()[0]

    # Forward pass through pytorch network.
    if use_caffe_processing:
        x = torch.from_numpy(caffe_net.blobs['data'].data)
        x = x[:, [2, 1, 0], :, :]
    else:
        assert isinstance(img_path, str)
        transform = get_transform(task_name=task_name, size=227)
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0)

    debug_probes = attach_debug_probes(torch_net, debug=debug)
    x.requires_grad_(True)
    torch_res = torch_net(x)

    # Backward passes.
    gradient = torch.ones_like(torch_res)
    torch_res.backward(gradient)

    caffe_net, caffe_res_back = cutils.net_backward(caffe_net,
                                                    caffe_end_blob,
                                                    gradient.cpu().data.numpy())
    assert len(caffe_res_back) == 1
    caffe_start_blob = caffe_res_back.keys()[0]

    # Check maximum difference.
    max_act_diff = np.abs(torch_res.cpu().data.numpy() - caffe_res[caffe_end_blob]).max()
    act_mag = np.abs(torch_res.cpu().data.numpy()).max()
    rel_max_act_diff = max_act_diff / act_mag
    mean_act_diff = np.abs(torch_res.cpu().data.numpy() - caffe_res[caffe_end_blob]).mean()
    max_grad_diff = np.abs(x.grad.cpu().data.numpy() - caffe_res_back[caffe_start_blob][:, [2, 1, 0], :, :]).max()
    grad_mag = np.abs(x.grad.cpu().data.numpy()).max()
    rel_max_grad_diff = max_grad_diff / grad_mag
    mean_grad_diff = np.abs(x.grad.cpu().data.numpy() - caffe_res_back[caffe_start_blob][:, [2, 1, 0], :, :]).mean()
    if plot_grad:
        f, ax = plt.subplots(1, 3, figsize=(3*4, 4))
        ax[0].imshow(np.transpose(normalize(x.grad.cpu().data.numpy()[0]), (1, 2, 0)))
        ax[1].imshow(np.transpose(normalize(caffe_res_back[caffe_start_blob][0, [2,1,0], :, :]), (1, 2, 0)))
        ax[2].imshow(np.transpose(normalize(x.grad.cpu().data.numpy()[0] - caffe_res_back[caffe_start_blob][0, [2,1,0],:, :]), (1, 2, 0)))
        ax[0].set_ylabel('{} ({})'.format(task_name, caffe_start_blob))
        ax[0].set_title('pytorch')
        ax[0].set_xlabel('use_caffe_processing {}'.format(use_caffe_processing))
        ax[1].set_title('caffe')
        ax[2].set_title('diff')
        ax[2].set_xlabel('Diff-Abs: {:.1e} Rel: {:.1e}'.format(max_grad_diff, rel_max_grad_diff))
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        plt.show()

    print('Max Act Diff ({}): {:.2e}'.format(caffe_end_blob, max_act_diff))
    print('Mean Act Diff ({}): {:.2e}'.format(caffe_end_blob, mean_act_diff))
    print('Max Act Mag ({}): {:.2e}'.format(caffe_end_blob, act_mag))
    print('Rel Max Act Diff ({}): {:.2e}'.format(caffe_end_blob, rel_max_act_diff))
    print('Max Grad Diff ({}): {:.2e}'.format(caffe_start_blob, max_grad_diff))
    print('Mean Grad Diff ({}): {:.2e}'.format(caffe_start_blob, mean_grad_diff))
    print('Max Grad Mag ({}): {:.2e}'.format(caffe_start_blob, grad_mag))
    print('Rel Max Grad Diff ({}): {:.2e}'.format(caffe_start_blob, rel_max_grad_diff))

    try:
        assert rel_max_act_diff < max_act_diff_threshold
        assert rel_max_grad_diff < max_grad_diff_threshold
    except:
        # TODO(ruthfong): Remove.
        # TODO(ruthfong): Figure out why backward gradients don't match AUDIO and EGOMOTION.
        import pdb; pdb.set_trace();
        pass

    if debug:
        for caffe_layer_name in caffe_net.blobs.keys()[::-1]:
            if learnable_classifier is not None and caffe_layer_name in learnable_classifier:
                layer_i,_ = learnable_classifier[caffe_layer_name]
                torch_layer_name = 'classifier.{}'.format(layer_i)
            elif caffe_layer_name in learnable_features:
                layer_i,_ = learnable_features[caffe_layer_name]
                torch_layer_name = 'features.{}'.format(layer_i)
            elif caffe_layer_name == 'data':
                torch_layer_name = ''
            else:
                import pdb; pdb.set_trace();
                assert False
            assert len(debug_probes[torch_layer_name].data) == 1

            if caffe_layer_name == 'data':
                act_diff = np.abs(debug_probes[torch_layer_name].data[0].cpu().data.numpy()
                                  - caffe_net.blobs[caffe_layer_name].data[:, [2, 1, 0], :, :])
                grad_diff = np.abs(debug_probes[torch_layer_name].data[0].grad.cpu().data.numpy()
                                   - caffe_net.blobs[caffe_layer_name].diff[:, [2, 1, 0], :, :])
            else:
                act_diff = np.abs(debug_probes[torch_layer_name].data[0].cpu().data.numpy()
                                  - caffe_net.blobs[caffe_layer_name].data)
                grad_diff = np.abs(debug_probes[torch_layer_name].data[0].grad.cpu().data.numpy()
                                  - caffe_net.blobs[caffe_layer_name].diff)
            act_mag = np.abs(debug_probes[torch_layer_name].data[0].grad.cpu().data.numpy()).max()
            grad_mag = np.abs(debug_probes[torch_layer_name].data[0].grad.cpu().data.numpy()).max()
            print('Layer - Caffe: {}\tPyTorch {}'.format(caffe_layer_name, torch_layer_name))
            print('Abs Diff - Act Max: {:.2e}\tMean: {:.2e}'.format(act_diff.max(), act_diff.mean()))
            print('Rel Diff - Act Max: {:.2e}\tMean: {:.2e}'.format(act_diff.max()/act_mag, act_diff.mean()/act_mag))
            print('Abs Diff - Grad Max: {:.2e}\tMean: {:.2e}'.format(grad_diff.max(), grad_diff.mean()))
            print('Rel Diff - Grad Max: {:.2e}\tMean: {:.2e}'.format(grad_diff.max()/grad_mag, grad_diff.mean()/grad_mag))

    # Save path.
    if not test_conversion:
        save_path = os.path.join(pytorch_models_dir,
                                 'alexnet-' + task_name + '.pt')
        torch.save(torch_net.state_dict(), save_path)
        print('Saved pytorch model to ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name',
                        type=str,
                        choices=TASK_NAMES,
                        default=PUZZLE)
    parser.add_argument('--caffe_models_dir',
                        type=str,
                        default=DEFAULT_CAFFE_MODELS_DIR)
    parser.add_argument('--max_act_diff_threshold',
                        type=float,
                        default=DEFAULT_REL_MAX_ACT_DIFF_THRESHOLD)
    parser.add_argument('--max_grad_diff_threshold',
                        type=float,
                        default=DEFAULT_REL_MAX_GRAD_DIFF_THRESHOLD)
    parser.add_argument('--use_caffe_processing',
                        action='store_true',
                        default=False)
    parser.add_argument('--test_conversion',
                        action='store_true',
                        default=False)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False)
    parser.add_argument('--plot_grad',
                        action='store_true',
                        default=False)
    parser.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()
    convert_caffe_to_pytorch(task_name=args.task_name,
                             caffe_models_dir=args.caffe_models_dir,
                             max_act_diff_threshold=args.max_act_diff_threshold,
                             max_grad_diff_threshold=args.max_grad_diff_threshold,
                             test_conversion=args.test_conversion,
                             use_caffe_processing=args.use_caffe_processing,
                             debug=args.debug,
                             plot_grad=args.plot_grad,
                             gpu=args.gpu)
