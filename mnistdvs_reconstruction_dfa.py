import argparse

import numpy as np
import torch
from snn.utils.misc import str2bool
from models.encoders import DFAEncoder

from utils.misc import get_decoder, make_results_dir
from utils.test_utils import get_acc_reconstruction_mnistdvs
import torchvision
from snn.optimizer.snnsgd import SNNSGD
from neurodata.load_data import create_dataloader
import tables
import tqdm
from utils.loss import EpropLoss

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--dataset', default=r"/datasets/mnist-dvs/mnist_dvs_events_new.hdf5")

    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--test_period', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--encoder_type', type=str, default='single_layer', choices=['no_hidden', 'single_layer', 'two_layers', 'mlp', 'lenet'])
    parser.add_argument('--decoder_type', type=str, default='conv')
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')
    parser.add_argument('--single_target', type=str, default='true', help='Use one target per class')
    parser.add_argument('--n_outputs_enc', type=int, default=256)
    parser.add_argument('--cut_period', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_dec', type=float, default=None)
    parser.add_argument('--kappa', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--lamda', type=float, default=0.)
    parser.add_argument('--r', type=float, default=0.1)
    parser.add_argument('--dt', type=int, default=25000)

    parser.add_argument('--results_path', type=str, default='', help='')

    args = parser.parse_args()

args.disable_cuda = str2bool(args.disable_cuda)
args.single_target = str2bool(args.single_target)
args.resume = False
args.results_path = make_results_dir('mnistdvs_reconstruction', args)
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
train_dl, test_dl = create_dataloader(dataset_path, batch_size=args.batch_size, size=input_size, classes=digits, sample_length_train=sample_length,
                                      sample_length_test=sample_length, dt=args.dt, polarity=True, ds=1, shuffle_test=True, num_workers=0)
### MNIST targets
mnist = torchvision.datasets.MNIST('../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
if args.single_target:
    mnist_loader = torch.utils.data.DataLoader(mnist, shuffle=False, batch_size=1)
    mnist_iter = list(iter(mnist_loader))
    targets_per_digit = [[img[0] for img in mnist_iter if img[1] == i][0] for i in digits]
else:
    mnist_loader = torch.utils.data.DataLoader(mnist, shuffle=True, batch_size=1)


r = (torch.ones([args.batch_size, 1, args.n_outputs_enc], requires_grad=False) * args.r).to(args.device)


### Make networks
if args.encoder_type == 'lenet':
    encoding_network = DFAEncoder(input_size,
                                  args.n_outputs_enc,
                                  batch_size=args.batch_size,
                                  Nhid_conv=[32, 64, 128],
                                  Nhid_mlp=[],
                                  kernel_size=[7],
                                  stride=[1],
                                  pool_size=[2, 1, 2],
                                  num_conv_layers=3,
                                  dropout=[0.2],
                                  num_mlp_layers=0,
                                  device=args.device
                                  ).to(args.device)
else:
    if args.encoder_type == 'single_layer':
        hidden_neurons = [1024]
    elif args.encoder_type == 'two_layers':
        hidden_neurons = [1024, 1024]
    elif args.encoder_type == 'mlp':
        hidden_neurons = [784, 1024, 1024]
    else:
        hidden_neurons = []
    encoding_network = DFAEncoder([np.prod(input_size)],
                                  args.n_outputs_enc,
                                  batch_size=args.batch_size,
                                  Nhid_conv=[],
                                  Nhid_mlp=hidden_neurons,
                                  num_mlp_layers=len(hidden_neurons),
                                  num_conv_layers=0,
                                  dropout=[0.],
                                  device=args.device
                                  ).to(args.device)


encoder_optimizer = SNNSGD([{'params': encoding_network.parameters(), 'ls': True, 'baseline': True}], lr=args.lr)

n_inputs_decoder = args.n_outputs_enc * tau_d
if args.lr_dec is None:
    args.lr_dec = args.lr
decoding_network, optimizer = get_decoder(args.decoder_type, args.device, args.lr_dec,
                                          T=T,
                                          in_features=n_inputs_decoder,
                                          hid_features=(args.n_outputs_enc * tau_d)//2,
                                          out_features=28 * 28,
                                          sigmoid=True
                                          )

print(encoding_network)
print(decoding_network)

loss_fn = torch.nn.BCELoss()
eprop_loss_fn = EpropLoss(loss_fn)
targets = torch.Tensor()

acc_best = 0

for epoch in range(args.n_epochs):
    train_iterator = iter(train_dl)
    test_iter = iter(test_dl)

    decoding_network.train()
    encoding_network.train()

    for inputs, label in tqdm.tqdm(train_iterator):
        if len(inputs) == args.batch_size:
            labels = torch.sum(label, dim=-1).argmax(-1).numpy()
            targets = torch.Tensor().to(args.device)

            ### Make targets
            for label in labels:
                if args.single_target:
                    target = targets_per_digit[np.where(digits == label)[0][0]].to(args.device)
                    targets = torch.cat((targets, target), dim=0)
                else:
                    mnist_iter = iter(mnist_loader)
                    target, target_lbl = next(mnist_iter)
                    while target_lbl != digits[label]:
                        target, target_lbl = next(mnist_iter)
                    targets = torch.cat((targets, target.to(args.device)))
            targets = targets.view([args.batch_size, -1])

            enc_loss = 0

            ### Reset period
            with torch.no_grad():
                length = encoding_network.out_layer.memory_length + 1
                refractory_sig = torch.zeros([args.batch_size, length] + input_size)
                for t in range(length):
                    enc_outputs, _ = encoding_network(refractory_sig[:, t].to(args.device))
                encoder_outputs = enc_outputs

            ### Forward pass
            encoding_network.detach_()
            for t in range(T):
                enc_outputs, s = encoding_network(inputs[:, t].to(args.device))
                encoder_outputs = torch.cat((encoder_outputs[:, -tau_d+1:], enc_outputs), dim=1)
                # Compute gradients
                eprop_loss = eprop_loss_fn(s, enc_outputs)
                eprop_loss.backward(retain_graph=(t+1) % args.cut_period)
                # Compute learning signal
                enc_loss += -loss_fn(s[-1].detach(), enc_outputs) + loss_fn(r, enc_outputs)
                if ((t+1) % args.cut_period) == 0:
                    encoding_network.detach_()

            ### Decoder computations
            decoder_outputs = decoding_network(encoder_outputs.unsqueeze(1))
            dec_loss = loss_fn(decoder_outputs, targets)
            dec_loss.backward()

            ### Gradients steps
            encoder_optimizer.step(dec_loss.detach() + args.beta * enc_loss)
            encoder_optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()

    if (epoch + 1) % args.test_period == 0:
        acc = get_acc_reconstruction_mnistdvs(encoding_network, decoding_network, test_iter, args, T, acc_best, tau_d, args.batch_size, digits)
        if acc > acc_best:
            acc_best = acc
        print('Epoch %d/%d' % (epoch + 1, args.n_epochs))
        print('Acc: %f, Best acc: %f' % (acc, acc_best))
