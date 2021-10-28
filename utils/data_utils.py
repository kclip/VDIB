import numpy as np
import torch
from scipy import special


def make_blob(n_neurons, T):
    # Create 'drifting blob'stimuli, as defined in [Chalk '17]
    min_range = - np.pi
    max_range = np.pi
    theta_blob = np.zeros([T])
    v = np.zeros([T])
    A = np.zeros([T])
    blob = np.zeros([T])
    eta = np.random.normal(0, 1, [T])

    inputs = np.zeros([n_neurons, T])
    bins = np.arange(min_range, max_range, (max_range - min_range) / n_neurons)

    A[0] = 0.5
    v[0] = 1
    theta_blob[0] = np.random.uniform(-np.pi, np.pi)
    blob[0] = A[0] * np.random.normal(theta_blob[0], 0.45)
    inputs[np.argmin(np.abs(bins - blob[0])), 0] = 1

    for t in range(1, T):
        theta_blob[t] = theta_blob[t - 1] + v[t - 1]
        v[t] = 0.9 * v[t - 1] - 0.14 * eta[t]
        A[t] = 0.99 * A[t - 1] + 0.01 * eta[t]

        blob[t] = A[t] * np.random.normal(theta_blob[t], 0.45)
        blob[t] = np.arctan2(np.sin(blob[t]), np.cos(blob[t]))

        inputs[np.argmin(np.abs(bins - blob[t])), t] = 1

    return torch.FloatTensor(inputs)


def get_possible_outputs(n_neurons):
    possible_outputs = []
    for i in range(n_neurons):
        for j in range(i, n_neurons):
            output = np.zeros(n_neurons)
            output[i] = 1
            output[j] = 1
            possible_outputs.append(output)
    return possible_outputs


def get_one_hot_encoding(array, n_neurons, possible_outputs):
    assert(torch.sum(array) == 2), 'Input array must have 2 entries equal to 1, found: %d' % torch.sum(array)
    array /= torch.max(array)
    one_hot = torch.zeros(special.comb(n_neurons, 1).astype(int) + special.comb(n_neurons, 2).astype(int))
    one_hot[[i for i in range(len(possible_outputs)) if np.array_equal(array, possible_outputs[i])]] = 1
    assert(torch.sum(one_hot) == 1), array
    return one_hot.unsqueeze(0)


def get_one_hot_index(array, possible_outputs):
    assert(torch.sum(array) <= 2), 'Input array must have at most 2 entries equal to 1, found: %d' % torch.sum(array)
    array /= torch.max(array)
    one_hot = torch.zeros([len(possible_outputs)])
    one_hot[[i for i in range(len(possible_outputs)) if np.array_equal(array.cpu(), possible_outputs[i])]] = 1

    assert(torch.sum(one_hot) == 1), print([arr for arr in possible_outputs if np.array_equal(array, arr)])
    return torch.argmax(one_hot).unsqueeze(0)


def make_cos_signal(n_neurons, T, f):
    inputs = np.zeros([n_neurons, T])
    bins = np.arange(-1, 1, 2 / n_neurons)

    phi = np.random.normal([1])

    for t in range(1, T):
        cos = np.cos(2 * np.pi * t / f + phi)
        inputs[np.argmin(np.abs(bins - cos)), t] = 1

    return torch.FloatTensor(inputs)


def make_test_signal(n_neurons, T, f):
    inputs = np.zeros([n_neurons, T])
    bins = np.arange(-1, 1, 2 / n_neurons)

    phi = np.random.normal([1])

    for t in range(1, T):
        cos = np.cos(2 * np.pi * t / f)
        inputs[np.argmin(np.abs(bins - cos)), t] = 1

    res = inputs
    # inputs_flat = np.zeros([n_neurons, T])
    # inputs_flat[n_neurons // 2, :] = 1
    # res = inputs + inputs_flat
    # res[res == 2.] = 1.

    return torch.FloatTensor(res)


### Test to remove

def make_moon_dataset(n_samples, T, noise, n_neuron_per_dim, res=100):
    '''
    Generates points from the Two Moons dataset, rescale them in [0, 1] x [0, 1] and encodes them using population coding
    '''

    from sklearn import datasets
    data = datasets.make_moons(n_samples=n_samples, noise=noise)

    c_intervals = res / max(n_neuron_per_dim - 1, 1)
    c = np.arange(0, res + c_intervals, c_intervals)

    data[0][:, 0] = (data[0][:, 0] - np.min(data[0][:, 0])) / (np.max(data[0][:, 0]) - np.min(data[0][:, 0]))
    data[0][:, 1] = (data[0][:, 1] - np.min(data[0][:, 1])) / (np.max(data[0][:, 1]) - np.min(data[0][:, 1]))

    binary_inputs = torch.zeros([len(data[0]), T, 2 * n_neuron_per_dim])
    binary_outputs = torch.zeros([len(data[0]), T, 1])

    for i, sample in enumerate(data[0]):
        rates_0 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, :n_neuron_per_dim] = torch.bernoulli(torch.tensor(rates_0).unsqueeze(0).repeat(T, 1))

        rates_1 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[1] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, n_neuron_per_dim:] = torch.bernoulli(torch.tensor(rates_1).unsqueeze(0).repeat(T, 1))

        binary_outputs[i, :] = data[1][i]

    return binary_inputs, binary_outputs, torch.FloatTensor(data[0]), torch.FloatTensor(data[1])


def make_moon_test(n_samples_per_dim, T, n_neuron_per_dim, res=100):
    '''
    Generates a grid of equally spaced points in [0, 1] x [0, 1] and encodes them as binary signals using population coding
    '''

    n_samples = n_samples_per_dim ** 2

    c_intervals = res / max(n_neuron_per_dim - 1, 1)
    c = np.arange(0, res + c_intervals, c_intervals)

    binary_inputs = torch.zeros([n_samples, T, 2 * n_neuron_per_dim])

    y, x = np.meshgrid(np.arange(n_samples_per_dim), np.arange(n_samples_per_dim))
    x = (x / n_samples_per_dim).flatten()
    y = (y / n_samples_per_dim).flatten()

    for i in range(n_samples):
        rates_0 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (x[i] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, :n_neuron_per_dim] = torch.bernoulli(torch.tensor(rates_0).unsqueeze(0).repeat(T, 1))

        rates_1 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (y[i] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, n_neuron_per_dim:] = torch.bernoulli(torch.tensor(rates_1).unsqueeze(0).repeat(T, 1))

    return binary_inputs, torch.FloatTensor(x), torch.FloatTensor(y)


def make_1d_signal(T=100, step_train=100, step_test=100, n_neuron_per_dim=10, res=100):
    ''''
    Generates a 1D signal, rescale the points in [0, 1] and encodes them using population coding
    '''

    x0 = np.arange(-1, 0, 1 / step_train)
    x1 = np.arange(1.5, 2.5, 1 / step_train)
    x2 = np.arange(4, 5, 1 / step_train)
    x_train = np.concatenate([x0, x1, x2])

    x_test = np.arange(-1, 5, 1 / step_test)

    def function(x):
        return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

    y_train = function(x_train)
    y_test = function(x_test)

    noise_std = 0.1
    noise_train = np.random.randn(*x_train.shape) * noise_std
    y_train = y_train + noise_train
    y_train = (y_train - np.min(y_test)) / (np.max(y_test) - np.min(y_test))
    x_train = (x_train - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    y_test = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    ### Population coding
    # Centers of the cosine basis
    c_intervals = 4 * (res / max(n_neuron_per_dim - 1, 1))
    c = np.arange(0, res + c_intervals, c_intervals / 4)

    x_train_bin = torch.zeros([len(x_train), T, n_neuron_per_dim])

    for i, sample in enumerate(zip(x_train, y_train)):
        rates_x = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        x_train_bin[i] = torch.bernoulli(torch.tensor(rates_x).unsqueeze(0).repeat(T, 1))

    x_test_bin = torch.zeros([len(x_test), T, n_neuron_per_dim])

    for i, sample in enumerate(zip(x_test, y_test)):
        rates_x = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        x_test_bin[i] = torch.bernoulli(torch.tensor(rates_x).unsqueeze(0).repeat(T, 1))

    return torch.FloatTensor(x_train), torch.FloatTensor(y_train), torch.FloatTensor(x_test), torch.FloatTensor(y_test), x_train_bin, x_test_bin


def encode_delta(x, deltas):
    if len(deltas) == 0:
        deltas = [deltas]
    x_delta = np.zeros(x.shape + (len(deltas), 2))
    for i in range(1, x.shape[1]):
        for j, delta in enumerate(deltas):
            x_delta[:, i, j, 0] = (x[:, i] - x[:, i-1]) > delta
            x_delta[:, i, j, 1] = (x[:, i] - x[:, i-1]) < -delta
    x_delta = torch.Tensor(x_delta).view(x.shape + (2 * len(deltas),))
    return x_delta


class CustomDataset(torch.utils.data.Dataset):
    '''
    Wrapper to create dataloaders from the synthetic datasets
    '''

    def __init__(self, data, target):
        self.data = data
        self.target = target
        super(CustomDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key], self.target[key]
