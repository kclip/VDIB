from __future__ import print_function
import tables
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_
from snn.utils import filters
import math
import numpy as np


class SNNLayer(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, synaptic_filter=filters.raised_cosine_pillow_08,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=0.5, device='cpu'):
        super(SNNLayer, self).__init__()

        self.device = device

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        ### Feedforward connections
        self.n_basis_feedforward = n_basis_feedforward
        self.feedforward_filter = synaptic_filter(tau_ff, self.n_basis_feedforward, mu).transpose(0, 1).to(self.device)
        self.feedforward_filter.requires_grad = False
        self.tau_ff = tau_ff

        ### Feedback connections
        self.n_basis_feedback = n_basis_feedback
        self.feedback_filter = synaptic_filter(tau_fb, self.n_basis_feedback, mu).transpose(0, 1).to(self.device)
        self.feedback_filter.requires_grad = False
        self.tau_fb = tau_fb

        self.ff_weights = torch.nn.parameter.Parameter(torch.Tensor(n_outputs, n_inputs, n_basis_feedforward))
        self.fb_weights = torch.nn.parameter.Parameter(torch.Tensor(n_outputs, n_basis_feedback))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(n_outputs))

        a = self.get_xavier()
        _no_grad_uniform_(self.ff_weights, -a, a)
        _no_grad_uniform_(self.fb_weights, -a, a)
        _no_grad_uniform_(self.bias, -a, a)


        self.spiking_history = torch.zeros([self.n_outputs, 2], requires_grad=True).to(self.device)

        self.potential = None

        ### Number of timesteps to keep in synaptic memory
        self.memory_length = max(self.tau_ff, self.tau_fb)


    def forward(self, input_history, target=None, no_update=False):
        ff_trace = self.compute_ff_trace(input_history)
        fb_trace = self.compute_fb_trace()

        self.potential = self.compute_ff_potential(ff_trace) + self.compute_fb_potential(fb_trace) + self.bias

        outputs = self.generate_spikes(target)
        if not no_update:
            self.spiking_history = self.update_spiking_history(outputs)

        # return logits
        return torch.sigmoid(self.potential), self.spiking_history[:, -1]

    def detach_(self):
        self.potential.detach_()
        self.spiking_history.detach_()

        self.ff_weights.detach_().requires_grad_()
        self.fb_weights.detach_().requires_grad_()
        self.bias.detach_().requires_grad_()

    def compute_ff_trace(self, input_history):
        if input_history.shape[-1] != self.feedforward_filter.shape[0]:
            return torch.matmul(input_history.flip(-1), self.feedforward_filter[:input_history.shape[-1]])
        else:
            return torch.matmul(input_history.flip(-1), self.feedforward_filter)

    def compute_ff_potential(self, ff_trace):
        return torch.sum(self.ff_weights * ff_trace, dim=(-1, -2))

    def compute_fb_trace(self):
        if self.spiking_history.shape[-1] != self.feedback_filter.shape[0]:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter[:self.spiking_history.shape[-1]])
        else:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter)

    def compute_fb_potential(self, fb_trace):
        return torch.sum(self.fb_weights * fb_trace, dim=(-1))

    def generate_spikes(self, target=None):
        if target is not None:
            return target
        else:
            try:
                outputs = torch.bernoulli(torch.sigmoid(self.potential)).to(self.device)
            except RuntimeError:
                print('Potential')
                print(self.potential)
                print('ff_weights', self.ff_weights.isnan().any())
                print('fb_weights', self.fb_weights.isnan().any())
                print('bias', self.bias.isnan().any())

            return outputs


    def update_spiking_history(self, new_spikes):
        with torch.no_grad():
            spiking_history = torch.cat((self.spiking_history[:, 1-self.memory_length:], torch.zeros([self.n_outputs, 1], requires_grad=True).to(self.device)), dim=-1)
            spiking_history[:, -1] = new_spikes

            return spiking_history


    def reset_weights(self):
        torch.nn.init.xavier_uniform_(self.fb_weights)
        torch.nn.init.xavier_uniform_(self.ff_weights)
        torch.nn.init.xavier_uniform_(self.bias)

    def get_xavier(self, gain=1.):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.ff_weights)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

        return a


class SNNLayerv2(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, batch_size, synaptic_filter=filters.raised_cosine_pillow_08,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=0.5, device='cpu'):
        super(SNNLayerv2, self).__init__()

        self.device = device

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batch_size = batch_size

        ### Feedforward connections
        self.n_basis_feedforward = n_basis_feedforward
        self.feedforward_filter = synaptic_filter(tau_ff, self.n_basis_feedforward, mu).transpose(0, 1).to(self.device)
        self.feedforward_filter.requires_grad = False
        self.tau_ff = tau_ff

        ### Feedback connections
        self.n_basis_feedback = n_basis_feedback
        self.feedback_filter = synaptic_filter(tau_fb, self.n_basis_feedback, mu)[0].to(self.device)
        self.feedback_filter.requires_grad = False
        self.tau_fb = tau_fb

        self.ff_synapses = torch.nn.ModuleList([torch.nn.Linear(n_inputs, n_outputs, bias=False) for _ in range(n_basis_feedforward)])
        # [torch.nn.init.uniform_(l.weight, -1/(n_inputs + n_outputs)**2,  1/(n_inputs + n_outputs)**2) for l in self.ff_synapses]
        # [torch.nn.init.uniform_(l.weight, -1/(n_inputs + n_outputs),  0) for l in self.ff_synapses]
        self.fb_synapse = torch.nn.Linear(n_outputs, n_outputs, bias=True)
        # torch.nn.init.uniform_(self.fb_synapse.weight, -1/(2*n_outputs)**2, 1/(2*n_outputs)**2)
        # torch.nn.init.uniform_(self.fb_synapse.bias, -1/(2*n_outputs)**2, 1/(2*n_outputs)**2)
        # torch.nn.init.uniform_(self.fb_synapse.weight, -1/(2*n_outputs), 0)
        # torch.nn.init.uniform_(self.fb_synapse.bias, -1/(2*n_outputs), 0)

        self.spiking_history = torch.zeros([self.batch_size, self.n_outputs, 2], requires_grad=True).to(self.device)

        self.potential = None

        ### Number of timesteps to keep in synaptic memory
        self.memory_length = max(self.tau_ff, self.tau_fb)



    def forward(self, input_history, target=None, no_update=False):
        ff_trace = self.compute_ff_trace(input_history)
        fb_trace = self.compute_fb_trace()

        ff_potential = self.compute_ff_potential(ff_trace)
        fb_potential = self.compute_fb_potential(fb_trace)
        self.potential = ff_potential + fb_potential

        outputs = self.generate_spikes(target)
        if not no_update:
            self.spiking_history = self.update_spiking_history(outputs)

        # return logits
        return torch.sigmoid(self.potential), self.spiking_history[:, :, -1]

    def detach_(self):
        self.potential.detach_()
        self.spiking_history.detach_()

        # [l.weight.detach_().requires_grad_() for l in self.ff_synapses]
        # self.fb_synapse.weight.detach_().requires_grad_()
        # self.fb_synapse.bias.detach_().requires_grad_()

    def compute_ff_trace(self, input_history):
        # input_history: shape = [n_batch, n_in, t]
        # feedforward filter: shape = [tau_ff, n_basis_ff]
        # res: shape = [[n_batch, n_in] * n_basis_ff]
        if input_history.shape[-1] != self.feedforward_filter.shape[0]:
            return [torch.matmul(input_history.flip(-1), self.feedforward_filter[:input_history.shape[-1], i]) for i in range(self.n_basis_feedforward)]
        else:
            return [torch.matmul(input_history.flip(-1), self.feedforward_filter[:input_history.shape[-1], i]) for i in range(self.n_basis_feedforward)]

    def compute_ff_potential(self, ff_trace):
        # ff_trace: shape = [[n_batch, n_in] * n_basis_ff]
        # ff_synapses: shape = [[n_out, n_in] * n_basis_ff]

        return torch.cat([self.ff_synapses[i](ff_trace[i]).unsqueeze(2) for i in range(self.n_basis_feedforward)], dim=-1).sum(dim=-1)

    def compute_fb_trace(self):
        # input_history: shape = [n_batch, n_out, t]
        # feedforward filter: shape = [tau_fb, 1]
        # res: shape = [n_batch, n_in]
        if self.spiking_history.shape[-1] != self.feedback_filter.shape[0]:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter[:self.spiking_history.shape[-1]])
        else:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter)

    def compute_fb_potential(self, fb_trace):
        return self.fb_synapse(fb_trace)


    def generate_spikes(self, target=None):
        if target is not None:
            return target
        else:
            try:
                outputs = torch.bernoulli(torch.sigmoid(self.potential)).to(self.device)
            except RuntimeError:
                print('Potential')
                print(self.potential)
                print('ff_weights', self.ff_weights.isnan().any())
                print('fb_weights', self.fb_weights.isnan().any())
                print('bias', self.bias.isnan().any())

            return outputs


    def update_spiking_history(self, new_spikes):
        with torch.no_grad():
            spiking_history = torch.cat((self.spiking_history[:, :, 1-self.memory_length:],
                                         torch.zeros([self.batch_size, self.n_outputs, 1], requires_grad=True).to(self.device)), dim=-1)
            spiking_history[:, :, -1] = new_spikes

            return spiking_history


    def get_xavier(self, gain=1.):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.fb_weights)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

        return a


class LayeredSNN(torch.nn.Module):
    def __init__(self, n_input_neurons, n_neurons_per_layer, n_output_neurons, synaptic_filter=filters.raised_cosine_pillow_08, n_basis_feedforward=[8],
                 n_basis_feedback=[1], tau_ff=[10], tau_fb=[10], mu=[0.5], device='cpu'):

        super(LayeredSNN, self).__init__()
        '''
        '''

        self.device = device

        ### Network parameters
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = np.sum(n_neurons_per_layer)
        self.n_hidden_layers = len(n_neurons_per_layer)
        self.n_output_neurons = n_output_neurons
        self.n_neurons = n_input_neurons + self.n_hidden_neurons + self.n_output_neurons

        if len(n_basis_feedforward) == 1:
            n_basis_feedforward = n_basis_feedforward * (1 + self.n_hidden_layers)
        if len(n_basis_feedback) == 1:
            n_basis_feedback = n_basis_feedback * (1 + self.n_hidden_layers)
        if len(tau_ff) == 1:
            tau_ff = tau_ff * (1 + self.n_hidden_layers)
        if len(tau_fb) == 1:
            tau_fb = tau_fb * (1 + self.n_hidden_layers)
        if len(mu) == 1:
            mu = mu * (1 + self.n_hidden_layers)

        self.hidden_layers = torch.nn.ModuleList()
        Nhid = [n_input_neurons] + n_neurons_per_layer

        for i in range(self.n_hidden_layers):
            self.hidden_layers.append(SNNLayer(Nhid[i], Nhid[i + 1], synaptic_filter=synaptic_filter, n_basis_feedforward=n_basis_feedforward[i],
                                               n_basis_feedback=n_basis_feedback[i], tau_ff=tau_ff[i], tau_fb=tau_fb[i], mu=mu[i], device=self.device))

        self.out_layer = SNNLayer(Nhid[-1], n_output_neurons, synaptic_filter=synaptic_filter, n_basis_feedforward=n_basis_feedforward[-1],
                                  n_basis_feedback=n_basis_feedback[-1], tau_ff=tau_ff[-1], tau_fb=tau_fb[-1], mu=mu[-1], device=self.device)

        self.training = None

    def forward(self, inputs_history, target=None, n_samples=1):
        probas_hidden = torch.Tensor().to(self.device)
        outputs_hidden = torch.Tensor().to(self.device)

        net_probas = torch.Tensor().to(self.device)
        net_outputs = torch.Tensor().to(self.device)

        for i in range(n_samples):
            if self.n_hidden_layers > 0:
                probas_hidden_tmp = torch.Tensor().to(self.device)
                outputs_hidden_tmp = torch.Tensor().to(self.device)

                proba_layer, layer_outputs = self.hidden_layers[0](inputs_history, target=None, no_update=n_samples - 1 - i)
                probas_hidden_tmp = torch.cat((probas_hidden_tmp, proba_layer.unsqueeze(0)))
                outputs_hidden_tmp = torch.cat((outputs_hidden_tmp, layer_outputs.unsqueeze(0)))

                for j in range(1, self.n_hidden_layers):
                    proba_layer, layer_outputs = self.hidden_layers[j](self.hidden_layers[j - 1].spiking_history, target=None, no_update=n_samples - 1 - i)
                    probas_hidden_tmp = torch.cat((probas_hidden_tmp, proba_layer.unsqueeze(0)))
                    outputs_hidden_tmp = torch.cat((outputs_hidden_tmp, layer_outputs.unsqueeze(0)))

                probas_hidden = torch.cat((probas_hidden, probas_hidden_tmp.unsqueeze(0)))
                outputs_hidden = torch.cat((outputs_hidden, outputs_hidden_tmp.unsqueeze(0)))

                probas_output_tmp, net_output_tmp = self.out_layer(self.hidden_layers[-1].spiking_history, target, no_update=n_samples - 1 - i)

            else:
                probas_output_tmp, net_output_tmp = self.out_layer(inputs_history, target, no_update=n_samples - 1 - i)
                probas_hidden = None
                outputs_hidden = None

            net_probas = torch.cat((net_probas, probas_output_tmp.unsqueeze(0)))
            net_outputs = torch.cat((net_outputs, net_output_tmp.unsqueeze(0)))

        return net_probas, net_outputs, probas_hidden, outputs_hidden

    ### Setters
    def reset_weights(self):
        for l in self.hidden_layers:
            l.reset_weights()

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def eval(self):
        self.training = False

    ### Misc
    def save(self, path=None):
        # todo
        return

    def import_weights(self, path):
        # todo
        return
