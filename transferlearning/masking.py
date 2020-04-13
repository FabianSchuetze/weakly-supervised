r"""
Defines the mask heads
"""
import torch

# TODO: Class agnostic is when out_features = 1
class SupervisedMask(torch.nn.Module):
    """ the convtional mask"""

    def __init__(self, in_features, dim_reduced, out_features):
        super(SupervisedMask, self).__init__()
        deconv = torch.nn.ConvTranspose2d(in_features, dim_reduced,
                                          kernel_size=(2, 2), stride=(2, 2))
        relu = torch.nn.ReLU()
        logits = torch.nn.Conv2d(dim_reduced, out_features,
                                 kernel_size=(1, 1), stride=(1, 1))
        self._module = torch.nn.Sequential(deconv, relu, logits)

    def forward(self, features):
        return self._module(features)


class WeaklySupervised(torch.nn.Module):
    """ the convtional mask"""

    def __init__(self, in_features, dim, n_classes):
        super(WeaklySupervised, self).__init__()
        self._dim = dim
        deconv = torch.nn.ConvTranspose2d(in_features, dim, kernel_size=(2, 2),
                                          stride=(2, 2))
        self._model = torch.nn.Sequential(deconv, torch.nn.ReLU())
        self._n_classes = n_classes

    def forward(self, features, weights):
        # import pdb; pdb.set_trace()
        out = self._model(features)
        weights = weights.view(self._n_classes, self._dim, 1, 1)
        return torch.nn.functional.conv2d(out, weights)

class TransferFunction(torch.nn.Module):
    """The transfer function from the paper"""

    def __init__(self, in_paras, out_paras, n_classes):
        super(TransferFunction, self).__init__()
        layer1 = torch.nn.Linear(in_paras, out_paras)
        layer2 = torch.nn.Linear(out_paras, out_paras)
        self._module = torch.nn.Sequential(layer1, torch.nn.ReLU(), layer2)
        self._n_classes = n_classes

    # TODO: How do deal with larger batch size?
    def forward(self, features):
        features = features.view(self._n_classes, -1)
        out = self._module(features)
        return out
