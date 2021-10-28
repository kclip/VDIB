import torch.nn as nn
import torch
import numpy as np
from collections import namedtuple
from itertools import chain
import warnings
from utils.misc import get_output_shape
from utils.activations import trainingHook
from snn.utils import filters

""""
LIF SNN 
Adapted from https://github.com/nmi-lab/decolle-public (code under GPLv3 License)
"""

dtype = torch.float32


class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])

    def __init__(self, layer, activation, alpha=.9, alpharp=.65, beta=.85, dim_hook=[], mode='eprop'):
        super(LIFLayer, self).__init__()
        self.base_layer = layer
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.alpharp = alpharp
        self.state = None
        self.activation = activation

        ### For DFA
        if len(dim_hook) > 0:
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            torch.nn.init.xavier_normal_(self.fixed_fb_weights, gain=1/np.prod(dim_hook))
            self.fixed_fb_weights.requires_grad = False
        else:
            self.fixed_fb_weights = None
        self.mode = mode


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

    def init_parameters(self):
        self.reset_parameters(self.base_layer)

    def forward(self, Sin_t, e):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + (1 - self.beta) * Sin_t
        P = self.alpha * state.P + (1 - self.alpha) * state.Q
        R = self.alpharp * state.R - (1 - self.alpharp) * state.S
        U = self.base_layer(P) + R

        S = self.activation(U, e, self.fixed_fb_weights, self.mode)

        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)

        return S, U


    def detach_(self):
        for s in self.state:
            s.detach_()


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


class TrainingHook(nn.Module):
    def __init__(self, dim_hook, mode):
        super(TrainingHook, self).__init__()

        # Feedback weights definition (FA feedback weights are handled in the FA_wrapper class)
        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False
        self.mode = mode

    def forward(self, input, e):
        return trainingHook(input, e, self.fixed_fb_weights, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.fixed_fb_weights.shape) + ')'


class SRMLayerWithDFA(nn.Module):
    def __init__(self, layer, activation, tau_ff=10, tau_fb=10, error_shape=[], hook_shape=[], mode='DFA'):
        super(SRMLayerWithDFA, self).__init__()
        self.base_layer = layer

        if mode is not None:
            self.hook = TrainingHook(hook_shape, mode)
            self.e = torch.nn.Parameter(torch.zeros(error_shape), requires_grad=False)
        else:
            self.hook = None
            self.e = None

        self.alpha = torch.nn.Parameter(torch.exp(torch.Tensor([-1/tau_ff])), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.exp(torch.Tensor([-1/tau_fb])), requires_grad=False)

        self.activation = activation
        self.ff_trace = None
        self.fb_trace = None
        self.U = None


    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
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

    def init_state(self):
        device = self.base_layer.weight.device
        self.ff_trace = torch.Tensor([0.]).to(device)
        self.fb_trace = torch.Tensor([0.]).to(device)

    def init_parameters(self):
        self.reset_parameters(self.base_layer)

    def forward(self, Sin_t):
        if self.ff_trace is None:
            self.init_state()

        self.ff_trace = self.alpha * self.ff_trace + Sin_t

        ff_potential = self.base_layer(self.ff_trace)
        self.U = ff_potential + self.fb_trace

        if isinstance(self.hook, TrainingHook):
            U_h = self.hook(self.U, self.e)
        else:
            U_h = self.U

        S = self.activation(U_h)
        self.fb_trace = self.beta * self.fb_trace - S.detach()
        return S

    def detach_(self):
        self.ff_trace.detach_()
        self.fb_trace.detach_()
        self.U.detach_()

        self.base_layer.weight.detach_().requires_grad_()
        self.base_layer.bias.detach_().requires_grad_()

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
