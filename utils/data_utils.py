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
