import argparse

import numpy as np
import torch
from snn.utils.misc import str2bool
from torchvision import datasets, transforms
from snn.optimizer.snnsgd import SNNSGD
from neurodata.load_data import create_dataloader
import tables

from models.encoders import DFAEncoder
from utils.misc import get_decoder, get_decoder_outputs, get_encoder_outputs, make_results_dir
from utils.test_utils import get_acc_reconstruction_mnistdvs


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--dataset', default=r"/datasets/mnist-dvs/mnist_dvs_events_new.hdf5")

    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--n_per_epoch', type=int, default=500)
    parser.add_argument('--n_examples_test', type=int, default=100)
    parser.add_argument('--test_period', type=int, default=1)

    parser.add_argument('--decoder_type', type=str, default='mlp')
    parser.add_argument('--decoding', type=str, default='')
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')
    parser.add_argument('--n_outputs_enc', type=int, default=256)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--r', type=float, default=0.3)
    parser.add_argument('--dt', type=int, default=10000)

    parser.add_argument('--resume', type=str, default='false', help='')
    parser.add_argument('--results_path', type=str, default='', help='')

    args = parser.parse_args()

args.disable_cuda = str2bool(args.disable_cuda)

args.resume = str2bool(args.resume)
args.results_path = make_results_dir('mnistdvs', args)

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

dataset_path = args.home + args.dataset
dataset = tables.open_file(dataset_path)
x_max = dataset.root.stats.train_data[1]
input_size = [2, 26, 26]
dataset.close()

sample_length = 2000000
T = int(sample_length / args.dt)
tau_d = T

digits = [i for i in range(10)]

# Create dataloaders
train_dl, test_dl = create_dataloader(dataset_path, batch_size=1, size=input_size, classes=digits, sample_length_train=sample_length,
                                      sample_length_test=sample_length, dt=args.dt, polarity=True, ds=1, shuffle_test=True, num_workers=0)


mnist = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

mnist_loader = torch.utils.data.DataLoader(mnist, shuffle=False, batch_size=1)
mnist_iter = list(iter(mnist_loader))
targets_per_digit = [[img[0] for img in mnist_iter if img[1] == i][0] for i in digits]


start_idx = 0
if args.resume:
    from collections import Counter
    previous_accs = np.load(args.results_path + '/accs.npy')
    start_idx = Counter(previous_accs)[0]

accs = np.zeros([5])

for j in range(start_idx, 5):
    hidden_neurons = [800]
    encoding_network = DFAEncoder([np.prod(input_size)],
                                  args.n_outputs_enc,
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
                                              T=T,
                                              in_features=n_inputs_decoder,
                                              hid_features=(args.n_outputs_enc * tau_d)//2,
                                              out_features=28 * 28,
                                              )

    print(encoding_network)
    print(decoding_network)

    loss_fn = torch.nn.BCELoss()

    targets = torch.Tensor()

    acc_best = 0

    for epoch in range(args.n_epochs):
        train_iterator = iter(train_dl)
        test_iter = iter(test_dl)

        decoding_network.train()
        encoding_network.train()


        for i in range(args.n_per_epoch):
            inputs, label = next(train_iterator)

            label = torch.sum(label, dim=-1).argmax(-1).numpy()
            if label in digits:
                target = targets_per_digit[np.where(digits == label)[0][0]].flatten().unsqueeze(0).to(args.device)
            else:
                raise RuntimeError

            enc_loss = 0
            encoder_outputs = torch.Tensor().to(args.device)

            with torch.no_grad():
                length = encoding_network.out_layer.out_layer.memory_length + 1
                refractory_sig = torch.zeros([1, length] + input_size)
                for t in range(length):
                    encoding_network(refractory_sig[:, t].to(args.device))

            for t in range(T):
                enc_outputs, enc_probas = encoding_network(inputs[:, t].to(args.device))

                encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, tau_d)

                enc_loss_tmp = torch.mean(enc_outputs * torch.log(1e-7 + enc_probas / args.r)
                                          + (1 - enc_outputs) * torch.log(1e-7 + (1. - enc_probas) / (1. - args.r))) / T
                enc_loss_tmp.backward()

                enc_loss += enc_loss_tmp.detach()

            decoder_outputs = get_decoder_outputs(decoding_network, encoder_outputs, args.decoding)
            dec_loss = loss_fn(decoder_outputs, target)
            dec_loss.backward()

            encoder_optimizer.step(dec_loss.detach() + args.beta * enc_loss)
            encoder_optimizer.zero_grad()

            optimizer.step()
            optimizer.zero_grad()


        if (epoch + 1) % args.test_period == 0:
            acc = get_acc_reconstruction_mnistdvs(encoding_network, decoding_network, test_iter, args, T, acc_best, j)
            if acc > acc_best:
                acc_best = acc
                accs[j] = acc

                np.save(args.results_path + '/accs.npy', accs)
                torch.save(encoding_network.state_dict(), args.results_path + '/encoding_network_trial_%d.pt' % j)
                torch.save(decoding_network.state_dict(), args.results_path + '/decoding_network_trial_%d.pt' % j)
            print('Epoch %d/%d' % (epoch + 1, args.n_epochs))
            print('Acc: %f' % acc)
