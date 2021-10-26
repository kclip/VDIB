import torch.nn as nn
import torch
import numpy as np
from collections import namedtuple
import warnings
from utils.misc import get_output_shape


""""
LIF SNN 
Adapted from https://github.com/nmi-lab/decolle-public (code under GPLv3 License)
"""

dtype = torch.float32


class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])

    def __init__(self, layer, activation, alpha=.9, alpharp=.65, beta=.85):
        super(LIFLayer, self).__init__()
        self.base_layer = layer
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.alpharp = alpharp
        self.state = None
        self.activation = activation

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        if hasattr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'):
            layer.weight.data[:] *= 0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3, 1e-3)
        else:
            warnings.warn('Unhandled layer type, not resetting parameters')

    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'):
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'):
            return layer.get_out_channels()
        else:
            raise Exception('Unhandled base layer type')

    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape,
                                    kernel_size=layer.kernel_size,
                                    stride=layer.stride,
                                    padding=layer.padding,
                                    dilation=layer.dilation)
        elif hasattr(layer, 'out_features'):
            return []
        elif hasattr(layer, 'get_out_shape'):
            return layer.get_out_shape()
        else:
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self, Sin_t):
        self.reset_parameters(self.base_layer)

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q.detach() + self.tau_s * Sin_t
        P = self.alpha * state.P.detach() + self.tau_m * state.Q
        R = self.alpharp * state.R - state.S.detach()
        U = self.base_layer(P) + R
        S = self.activation(U)

        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)

        return S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if hasattr(layer, 'out_channels'):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features


    def get_device(self):
        return self.base_layer.weight.device
