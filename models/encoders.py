import torch
from snn.models.base import SNNLayer
import numpy as np
from utils.activations import eprop_sigmoid
from models.LIF_base import LIFLayer
from torch import nn
from utils.misc import get_output_shape
from snn.utils import filters

""""
Parts of this code are adapted from 
https://github.com/nmi-lab/decolle-public (code under GPLv3 License)
https://github.com/ChFrenkel/DirectRandomTargetProjection (code under Apache v2.0 license)
"""


class DFAEncoder(torch.nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 batch_size,
                 mode='eprop',
                 Nhid_conv=[1],
                 Nhid_mlp=[128],
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 alpha=[.9],
                 beta=[.85],
                 alpharp=[.65],
                 tau_e=10,
                 activation=eprop_sigmoid,
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 lif_layer_type=LIFLayer,
                 with_bias=True,
                 dropout=[0.],
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
        if len(dropout) == 1:
            dropout = dropout * self.num_layers
        self.dropout = dropout


        super(DFAEncoder, self).__init__()

        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.mode = mode
        self.tau_e = tau_e

        if self.mode == 'eprop':
            self.e = torch.zeros([batch_size, output_shape], requires_grad=False).to(device)
        else:
            self.e = None

        self.LIF_layers = torch.nn.ModuleList()

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
            base_layer = nn.Conv2d(Nhid_conv[i], Nhid_conv[i + 1], kernel_size[i], stride[i], padding[i], bias=with_bias)

            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   alpha=alpha[i],
                                   beta=beta[i],
                                   alpharp=alpharp[i],
                                   dim_hook=[output_shape, Nhid_conv[i + 1], feature_height, feature_width],
                                   mode=self.mode
                                   )
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            self.LIF_layers.append(layer)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            self.pool_layers.append(pool)
            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            self.dropout_layers.append(dropout_layer)

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
                                   alpharp=alpharp[i],
                                   dim_hook=[output_shape, Nhid_mlp[i + 1]],
                                   mode=self.mode
                                   )

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            self.dropout_layers.append(dropout_layer)

        self.hidden_hist = None
        self.out_layer = SNNLayer(Nhid_mlp[-1], output_shape, batch_size, synaptic_filter=filter, n_basis_feedforward=8,
                                    n_basis_feedback=1, tau_ff=tau_e, tau_fb=tau_e, mu=0.5, device=device)

    def forward(self, inputs, targets=None):
        i = 0
        s_out = []
        if self.num_conv_layers == 0:
            inputs = inputs.view(inputs.size(0), -1)

        # Forward propagates the signal through all layers
        for lif, pool, do in zip(self.LIF_layers, self.pool_layers, self.dropout_layers):
            s, u = lif(inputs, self.e)
            s_out.append(s)
            s_p = pool(s)
            inputs = s_p.detach()
            i += 1
            if i == self.num_conv_layers:
                inputs = inputs.view(inputs.size(0), -1)

        if self.hidden_hist is not None:
            hidden_hist = torch.cat((self.hidden_hist[:, :, 1-self.out_layer.memory_length:], inputs.unsqueeze(2)), dim=-1)
        else:
            hidden_hist = inputs.unsqueeze(2)

        net_probas, net_outputs = self.out_layer(hidden_hist, targets)
        s_out.append(net_probas.unsqueeze(1))

        self.e.data.copy_(net_outputs.data.detach() - net_probas.data.detach())
        self.hidden_hist = hidden_hist

        return net_outputs.detach().unsqueeze(1), s_out


    def detach_(self):
        for l in self.LIF_layers:
            l.detach_()
        self.hidden_hist.detach_()
        self.out_layer.detach_()
