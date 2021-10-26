import torch
from snn.models.SNN import LayeredSNN
import numpy as np
from utils.activations import smooth_step
from models.LIF_base import LIFLayer
from torch import nn
from utils.activations import trainingHook
from utils.misc import get_output_shape
from snn.utils import filters

""""
Parts of this code are adapted from 
https://github.com/nmi-lab/decolle-public (code under GPLv3 License)
https://github.com/ChFrenkel/DirectRandomTargetProjection (code under Apache v2.0 license)
"""

class gaussian_encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_neurons, n_outputs_enc, device='cpu') -> None:
        super(gaussian_encoder, self).__init__()

        self.network = LayeredSNN(input_size, hidden_neurons[:-1], hidden_neurons[-1], device=device)
        self.mean = torch.nn.Linear(hidden_neurons[-1], n_outputs_enc)
        self.var = torch.nn.Linear(hidden_neurons[-1], n_outputs_enc)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, inputs, n_samples=1):
        _, _, probas_hidden, outputs_hidden = self.network(inputs, n_samples=n_samples)

        mu, logvar = self.mean(self.network.out_layer.potential), self.var(self.network.out_layer.potential)

        return self.reparameterize(mu, logvar), mu, logvar, probas_hidden, outputs_hidden


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


class FA_wrapper(nn.Module):
    ''''
    From https://github.com/ChFrenkel/DirectRandomTargetProjection/
    '''
    def __init__(self, module, dim, stride=None, padding=None):
        super(FA_wrapper, self).__init__()
        self.module = module
        self.stride = stride
        self.padding = padding
        self.output_grad = None
        self.x_shape = None

        # FA feedback weights definition
        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
        self.reset_weights()

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape
            x, u = self.module(x)
            x.register_hook(self.FA_hook_post)
            return x, u
        else:
            return self.module(x)

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            if isinstance(self.module.base_layer, torch.nn.Linear):
                return self.output_grad.mm(self.fixed_fb_weights)
            elif isinstance(self.module.base_layer, torch.nn.Conv2d):
                return torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad, self.stride, self.padding)
            else:
                raise NameError("=== ERROR: layer type " + str(self.self.module.base_layer) + " is not supported in FA wrapper")
        else:
            return grad

    def FA_hook_post(self, grad):
        self.output_grad = grad
        return grad


class DFAEncoder(torch.nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 mode='DFA',
                 Nhid_conv=[1],
                 Nhid_mlp=[128],
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 alpha=[.9],
                 beta=[.85],
                 alpharp=[.65],
                 tau_d=10,
                 activation=smooth_step,
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 lif_layer_type=LIFLayer,
                 with_bias=True,
                 filter=filters.raised_cosine_pillow_08,
                 device='cpu'
                 ):


        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        self.num_layers = num_conv_layers + num_mlp_layers

        if len(kernel_size) == 1:
            kernel_size = kernel_size * num_conv_layers
        if len(stride) == 1:
            stride = stride * num_conv_layers
        if len(pool_size) == 1:
            pool_size = pool_size * num_conv_layers
        if len(Nhid_conv) == 1:
            Nhid_conv = Nhid_conv * num_conv_layers
        if len(Nhid_mlp) == 1:
            Nhid_mlp = Nhid_mlp * num_mlp_layers
        if len(alpha) == 1:
            alpha = alpha * self.num_layers
        if len(alpharp) == 1:
            alpharp = alpharp * self.num_layers
        if len(beta) == 1:
            beta = beta * self.num_layers

        super(DFAEncoder, self).__init__()

        self.pool_layers = nn.ModuleList()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.mode = mode

        if (self.mode == 'DFA') or (mode == 'DRTP'):
            self.e = torch.zeros([1, output_shape], requires_grad=False).to(device)
        else:
            self.e = None

        self.LIF_layers = torch.nn.ModuleList()
        self.hooks = torch.nn.ModuleList()


        if num_conv_layers > 0:
            # Computing padding to preserve feature size
            padding = (np.array(kernel_size) - 1) // 2

            feature_height = input_shape[1]
            feature_width = input_shape[2]
            Nhid_conv = [input_shape[0]] + Nhid_conv

        # Creates LIF convolutional layers
        for i in range(num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = nn.Conv2d(Nhid_conv[i], Nhid_conv[i + 1], kernel_size[i], stride[i], padding[i], bias=with_bias)

            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   alpha=alpha[i],
                                   beta=beta[i],
                                   alpharp=alpharp[i]
                                   )
            pool = nn.MaxPool2d(kernel_size=pool_size[i])

            if (self.mode == 'DFA') or (mode == 'DRTP'):
                self.LIF_layers.append(layer)
            elif self.mode == 'FA':
                self.LIF_layers.append(FA_wrapper(layer, base_layer.weight.shape, stride=stride[i], padding=padding[i]))
            self.hooks.append(TrainingHook([output_shape, Nhid_conv[i + 1], feature_height, feature_width], self.mode))
            self.pool_layers.append(pool)

        if num_conv_layers > 0:
            mlp_in = int(feature_height * feature_width * Nhid_conv[-1])
            Nhid_mlp = [mlp_in] + Nhid_mlp
        else:
            Nhid_mlp = input_shape + Nhid_mlp

        # Creates LIF linear layers
        for i in range(num_mlp_layers):
            base_layer = nn.Linear(Nhid_mlp[i], Nhid_mlp[i + 1])
            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   alpha=alpha[i],
                                   beta=beta[i],
                                   alpharp=alpharp[i]
                                   )

            if (self.mode == 'DFA') or (mode == 'DRTP'):
                self.LIF_layers.append(layer)
            elif self.mode == 'FA':
                self.LIF_layers.append(FA_wrapper(layer, base_layer.weight.shape))
            self.hooks.append(TrainingHook([output_shape, Nhid_mlp[i+1]], self.mode))
            self.pool_layers.append(nn.Sequential())


        self.hidden_hist = None
        self.out_layer = LayeredSNN(Nhid_mlp[-1], [], output_shape, tau_ff=[tau_d], tau_fb=[tau_d], synaptic_filter=filter, device=device)

    def forward(self, inputs, targets=None):
        i = 0
        # Forward propagates the signal through all layers

        for lif, hook, pool in zip(self.LIF_layers, self.hooks, self.pool_layers):
            if i == self.num_conv_layers:
                inputs = inputs.view(inputs.size(0), -1)

            _, u = lif(inputs)
            u_p = pool(u)
            s_ = self.activation(u_p)
            s_ = hook(s_, self.e)

            inputs = s_
            i += 1

        if self.hidden_hist is not None:
            hidden_hist = torch.cat((self.hidden_hist[:, 1-self.out_layer.out_layer.memory_length:], inputs.flatten().unsqueeze(1)), dim=-1)
        else:
            hidden_hist = inputs.flatten().unsqueeze(1)

        net_probas, net_outputs, _, _ = self.out_layer(hidden_hist, targets, n_samples=1)

        if self.mode == 'DFA':
            self.e.data.copy_(net_outputs.data - net_probas.data)
        elif self.mode == 'DRTP':
            self.e.data.copy_(net_outputs.data)

        self.hidden_hist = hidden_hist.detach()
        return net_outputs.detach(), net_probas
