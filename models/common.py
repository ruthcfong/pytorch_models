r"""
This module defines common code for the backpropagation methods.

Copied from https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/common.py
"""

import torch
import weakref
from collections import OrderedDict

__all__ = [
    'attach_debug_probes',
]


class Patch(object):
    """Patch a callable in a module."""

    @staticmethod
    def resolve(target):
        """Resolve a target into a module and an attribute.
        The function resolves a string such as ``'this.that.thing'`` into a
        module instance `this.that` (importing the module) and an attribute
        `thing`.
        Args:
            target (str): target string.
        Returns:
            tuple: module, attribute.
        """
        target, attribute = target.rsplit('.', 1)
        components = target.split('.')
        import_path = components.pop(0)
        target = __import__(import_path)
        for comp in components:
            import_path += '.{}'.format(comp)
            __import__(import_path)
            target = getattr(target, comp)
        return target, attribute

    def __init__(self, target, new_callable):
        """Patch a callable in a module.
        Args:
            target (str): path to the callable to patch.
            callable (fun): new callable.
        """
        target, attribute = Patch.resolve(target)
        self.target = target
        self.attribute = attribute
        self.orig_callable = getattr(target, attribute)
        setattr(target, attribute, new_callable)

    def __del__(self):
        self.remove()

    def remove(self):
        """Remove the patch."""
        if self.target is not None:
            setattr(self.target, self.attribute, self.orig_callable)
        self.target = None
def _wrap_in_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


class _InjectContrast(object):
    def __init__(self, contrast, non_negative):
        self.contrast = contrast
        self.non_negative = non_negative

    def __call__(self, grad):
        assert grad.shape == self.contrast.shape
        delta = grad - self.contrast
        if self.non_negative:
            delta = delta.clamp(min=0)
        return delta


class _Catch(object):
    def __init__(self, probe):
        self.probe = weakref.ref(probe)

    def _process_data(self, data):
        if not self.probe():
            return
        p = self.probe()
        assert isinstance(data, list)
        p.data = data
        for i, x in enumerate(p.data):
            x.requires_grad_(True)
            x.retain_grad()
            if len(p.contrast) > i and p.contrast[i] is not None:
                injector = _InjectContrast(
                    p.contrast[i], p.non_negative_contrast)
                x.register_hook(injector)


class _CatchInputs(_Catch):
    def __call__(self, module, input):
        self._process_data(_wrap_in_list(input))


class _CatchOutputs(_Catch):
    def __call__(self, module, input, output):
        self._process_data(_wrap_in_list(output))


class Probe(object):
    """Probe for a layer.
    A probe attaches to a given :class:`torch.nn.Module` instance.
    While attached, the object records any data produced by the module along
    with the corresponding gradients. Use :func:`remove` to remove the probe.
    Examples:
        .. code:: python
            module = torch.nn.ReLU
            probe = Probe(module)
            x = torch.randn(1, 10)
            y = module(x)
            z = y.sum()
            z.backward()
            print(probe.data[0].shape)
            print(probe.data[0].grad.shape)
    """

    def __init__(self, module, target='input'):
        """Create a probe attached to the specified module.
        The probe intercepts calls to the module on the way forward, capturing
        by default all the input activation tensor with their gradients.
        The activation tensors are stored as a sequence :attr:`data`.
        Args:
            module (torch.nn.Module): Module to attach.
            target (str): Choose from ``'input'`` or ``'output'``. Use
                ``'output'`` to intercept the outputs of a module
                instead of the inputs into the module. Default: ``'input'``.
        .. Warning:
            PyTorch module interface (at least until 1.1.0) is partially
            broken. In particular, the hook functionality used by the probe
            work properly only for atomic module, not for containers such as
            sequences or for complex module that run several functions
            internally.
        """
        self.module = module
        self.data = []
        self.target = target
        self.hook = None
        self.contrast = []
        self.non_negative_contrast = False
        if hasattr(self.module, "inplace"):
            self.inplace = self.module.inplace
            self.module.inplace = False
        if self.target == 'input':
            self.hook = module.register_forward_pre_hook(_CatchInputs(self))
        elif self.target == 'output':
            self.hook = module.register_forward_hook(_CatchOutputs(self))
        else:
            assert False

    def __del__(self):
        self.remove()

    def remove(self):
        """Remove the probe."""
        if self.module is not None:
            if hasattr(self.module, "inplace"):
                self.module.inplace = self.inplace
            self.hook.remove()
            self.module = None


def attach_debug_probes(model, debug=False):
    r"""
    Returns an :class:`collections.OrderedDict` of :class:`Probe` objects for
    all modules in the model if :attr:`debug` is ``True``; otherwise, returns
    ``None``.
    Args:
        model (:class:`torch.nn.Module`): a model.
        debug (bool, optional): if True, return an OrderedDict of Probe objects
            for all modules in the model; otherwise returns ``None``.
            Default: ``False``.
    Returns:
        :class:`collections.OrderedDict`: dict of :class:`Probe` objects for
            all modules in the model.
    """
    if not debug:
        return None

    debug_probes = OrderedDict()
    for module_name, module in model.named_modules():
        debug_probe_target = "input" if module_name == "" else "output"
        debug_probes[module_name] = Probe(
            module, target=debug_probe_target)
    return debug_probes
