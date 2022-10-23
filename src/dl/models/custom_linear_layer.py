import torch
from torch import nn


class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, weights):
        super().__init__()
        self.size_in, self.size_out = weights.shape[0], weights.shape[-1]
        weights = torch.Tensor(self.size_in, self.size_out)
        self.weights = nn.Parameter(weights.T)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(self.size_out)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        w_times_x = torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b