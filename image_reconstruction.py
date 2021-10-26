import argparse

import numpy as np
import torch
from torchvision import datasets, transforms
from snn.optimizer.snnsgd import SNNSGD
from snn.utils.misc import str2bool

from models.encoders import DFAEncoder
from utils.misc import get_decoder, get_decoder_outputs, get_encoder_outputs, get_encoded_image, make_results_dir
from utils.test_utils import get_acc_reconstruction

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--n_examples_train', type=int, default=200000)
    parser.add_argument('--n_examples_test', type=int, default=1000)
    parser.add_argument('--test_period', type=int, default=500)

    parser.add_argument('--decoder_type', type=str, default='mlp')
    parser.add_argument('--encoding', type=str, default='rate')
    parser.add_argument('--decoding', type=str, default='')
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')
    parser.add_argument('--n_outputs_enc', type=int, default=256)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--r', type=float, default=0.3)
    parser.add_argument('--T', type=int, default=30)

    parser.add_argument('--resume', type=str, default='false', help='')
    parser.add_argument('--results_path', type=str, default='', help='')

    args = parser.parse_args()

args.resume = str2bool(args.resume)
args.results_path = make_results_dir('mnist', args)

args.disable_cuda = str2bool(args.disable_cuda)
args.single_target = str2bool(args.single_target)

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


size = 28
input_size = [size ** 2]

hidden_neurons = [600]
tau_d = args.T
digits = [i for i in range(10)]

transform = transforms.Compose([transforms.Resize([size, size]), transforms.ToTensor()])

mnist_train = datasets.MNIST('../data', train=True, download=True,
                             transform=transform)
mnist_test = datasets.MNIST('../data', train=False,
                            transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, shuffle=True, batch_size=1)
test_loader = torch.utils.data.DataLoader(mnist_test, shuffle=True, batch_size=1)


start_idx = 0
if args.resume:
    from collections import Counter
    previous_accs = np.load(args.results_path + '/accs.npy')
    start_idx = Counter(previous_accs)[0]

accs = np.zeros([5])

for i in range(start_idx, 5):
    encoding_network = DFAEncoder(input_size,
                                  args.n_outputs_enc,
                                  mode='DFA',
                                  Nhid_conv=[],
                                  Nhid_mlp=hidden_neurons,
                                  num_mlp_layers=len(hidden_neurons),
                                  num_conv_layers=0,
                                  device=args.device
                                  ).to(args.device)

    encoder_optimizer = SNNSGD([{'params': encoding_network.parameters(), 'ls': True, 'baseline': True}],
                               lr=args.lr)

    if args.decoding == 'rate':
        n_inputs_decoder = args.n_outputs_enc
    else:
        n_inputs_decoder = args.n_outputs_enc * tau_d

    decoding_network, optimizer = get_decoder(args.decoder_type, args.device, args.lr,
                                              T=args.T,
                                              in_features=n_inputs_decoder,
                                              hid_features=(args.n_outputs_enc * tau_d)//2,
                                              out_features=input_size[0],
                                              )

    print(encoding_network)
    print(decoding_network)

    loss_fn = torch.nn.BCELoss()

    train_iter = iter(train_loader)
    learning_signal = torch.Tensor([0.])
    targets = torch.Tensor()

    acc_best = 0

    for ite in range(args.n_examples_train):
        encoding_network.train()
        decoding_network.train()

        inputs, target, _ = get_encoded_image(train_iter, train_loader, digits, args.encoding, args.T)

        if (ite + 1) % args.test_period == 0:
            acc = get_acc_reconstruction(encoding_network, decoding_network, test_loader, args, digits, size, acc_best, i)

            if acc > acc_best:
                acc_best = acc
                accs[i] = acc
                np.save(args.results_path + '/accs.npy', accs)

            print(acc)


        decoding_network.train()
        encoding_network.train()

        inputs = inputs.unsqueeze(0)
        target = target.to(args.device)

        enc_loss = 0
        encoder_outputs = torch.Tensor().to(args.device)

        with torch.no_grad():
            length = encoding_network.out_layer.out_layer.memory_length + 1
            refractory_sig = torch.zeros([1, input_size[0], length])
            for t in range(length):
                encoding_network(refractory_sig[:, :, t].to(args.device))

        for t in range(args.T):
            enc_outputs, enc_probas = encoding_network(inputs[:, :, t].to(args.device))

            encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, tau_d)

            enc_loss_tmp = torch.mean(enc_outputs * torch.log(1e-7 + enc_probas / args.r)
                                      + (1 - enc_outputs) * torch.log(1e-7 + (1. - enc_probas) / (1. - args.r))) / args.T
            enc_loss_tmp.backward()

            enc_loss += enc_loss_tmp.detach()

        decoder_outputs = get_decoder_outputs(decoding_network, encoder_outputs, args.decoding)
        dec_loss = loss_fn(decoder_outputs, target)

        dec_loss.backward()

        encoder_optimizer.step(dec_loss.detach() + args.beta * enc_loss)
        encoder_optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()
