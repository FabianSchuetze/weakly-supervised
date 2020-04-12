r"""
Defines the mask heads
"""
from typing import List
import torch

class SupervisedMask(torch.nn.Module):
    """ the convtional mask"""

    def __init__(self, in_features, dim_reduced, out_features):
        super(SupervisedMask, self).__init__()
        self._deconv = torch.nn.ConvTranspose2d(in_features, dim_reduced,
                                                kernel_size=(2, 2),
                                                stride=(2, 2))
        self._logits = torch.nn.Conv2d(dim_reduced, out_features,
                                       kernel_size=(1, 1),
                                       stride=(1, 1))
    def forward(self, features):
        out = self._deconv(features)
        out = torch.nn.functional.relu(out)
        out = self._logits(out)
        return out

class WeaklySupervised(torch.nn.Module):
    """ the convtional mask"""

    def __init__(self, in_features, dim_reduced, out_features):
        super(WeaklySupervised, self).__init__()
        deconv = torch.nn.ConvTranspose2d(in_features, dim_reduced,
                                          kernel_size=(2, 2), stride=(2, 2))
        relu = torch.nn.ReLU
        logits = torch.nn.Conv2d(dim_reduced, out_features,
                                 kernel_size=(1, 1), stride=(1, 1))
        self._model = torch.nn.Sequential(deconv, relu, logits)

    def _disable_gradients(self):
        for i in self._model.parameters():
            i.requires_grad = False

    def _set_parameters(self, weights):
        for existing, new_weights in zip(self._model.parameters(), weights):
            existing.data = new_weights

    def forward(self, features, weights: List[torch.tensor]):
        self._set_parameters(weights)
        out = self._deconv(features)
        out = torch.nn.functional.relu(out)
        out = self._logits(out)
        return out

class TransferFunction(torch.nn.Module):
    """The transfer function from the paper"""

    def __init__(self, in_paras, individual_paras: List[int]):
        super(TransferFunction, self).__init__()
        total_paras = sum(individual_paras)
        self._paras = individual_paras
        self._layer = torch.nn.Linear(in_paras, total_paras)

    def forward(self, features):
        out = self._layer(features)
        outputs = []
        for idx in range(len(self._paras)):
            if idx < len(self._paras) - 1:
                outputs.append(out[self._paras[idx], self._paras[idx+1]])
            else:
                outputs.append(out[self._paras[idx]:])
        return outputs
