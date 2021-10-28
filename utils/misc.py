import fnmatch
import numpy as np
import os
import time
import torch
from snn.models.SNN import BinarySNN
from models.decoders import conv, mlp, linear
# from models.encoders import gaussian_encoder


def get_decoder(decoder_type, device, lr, **kwargs):
    if decoder_type == 'linear':
        decoding_network = linear(**kwargs).to(device)
    elif decoder_type == 'mlp':
        decoding_network = mlp(**kwargs).to(device)
    elif decoder_type == 'conv':
        decoding_network = conv(**kwargs).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(decoding_network.parameters(), lr=lr)

    return decoding_network, optimizer


def get_output_shape(input_shape, kernel_size=[3,3], stride = [1,1], padding=[1,1], dilation=[0,0], output_padding=[0,0]):
    if not hasattr(kernel_size, '__len__'):
        kernel_size = [kernel_size, kernel_size]
    if not hasattr(stride, '__len__'):
        stride = [stride, stride]
    if not hasattr(padding, '__len__'):
        padding = [padding, padding]
    if not hasattr(output_padding, '__len__'):
        padding = [output_padding, output_padding]
    if not hasattr(dilation, '__len__'):
        dilation = [dilation, dilation]
    im_height = input_shape[-2]
    im_width = input_shape[-1]
    height = int((im_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0]-1) + output_padding[0] + 1)
    width = int((im_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1]-1) + output_padding[1] + 1)
    return [height, width]


def get_acc(preds, labels, batch_size):
    with torch.no_grad():
        acc = torch.sum(preds == torch.sum(labels.cpu(), dim=-1).argmax(dim=1)).float() / batch_size

        return acc


def get_encoded_image(iterator, loader, digits, encoding, T, targets=None):
    try:
        x, label = next(iterator)
        while label not in digits:
            x, label = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        x, label = next(iterator)

    if targets is not None:
        target = targets[np.where(digits == label.numpy()[0])[0][0]].flatten().unsqueeze(0)
    else:
        target = x.flatten().unsqueeze(0)

    if encoding == 'rate':
        inputs = torch.bernoulli(x.flatten().unsqueeze(1).repeat(1, T))
    elif encoding == 'time':
        times = T - torch.round(x.flatten() * (T - 1))
        inputs = torch.zeros([len(x.flatten()), T])
        inputs[[i for i in range(len(times)) if times[i] != T], [t for t in times if t != T]] = 1

    return inputs, target, label


def make_results_dir(exp, args):
    if args.resume:
        results_path = args.results_path
    else:
        prelist = np.sort(fnmatch.filter(os.listdir(os.getcwd() + "\\results"), '[0-9][0-9][0-9]__*'))
        if len(prelist) == 0:
            expDirN = "001"
        else:
            expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

        results_path = time.strftime(os.getcwd() + '/results/' + expDirN + "__" + "%d-%m-%Y"
                                     + "_" + exp + '_' + args.decoder_type + '_beta_' + str(args.beta)
                                     + '_nenc_' + str(args.n_outputs_enc) + '__dt__' + str(args.dt),
                                     time.localtime())
        os.makedirs(results_path)

    return results_path
