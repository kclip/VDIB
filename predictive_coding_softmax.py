import argparse
import numpy as np
import torch
from snn.utils.misc import str2bool
from snn.optimizer.snnsgd import SNNSGD

from utils.misc import get_decoder, get_decoder_outputs, get_encoder_outputs, make_results_dir
from utils.test_utils import get_acc_predictive_softmax
from utils.data_utils import make_blob, get_possible_outputs, get_one_hot_index
from models.encoders import DFAEncoder


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--results', default=r"/home/results")
    parser.add_argument('--dataset', default=r"mnist_dvs")
    parser.add_argument('--n_examples_train', type=int, default=50000)
    parser.add_argument('--n_examples_test', type=int, default=100)
    parser.add_argument('--test_period', type=int, default=2000)

    parser.add_argument('--decoder_type', type=str, default='linear')
    parser.add_argument('--decoding', type=str, default='')

    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--T_test', type=int, default=1000)
    parser.add_argument('--delta', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--r', type=float, default=0.2)

    parser.add_argument('--resume', type=str, default='false', help='')
    parser.add_argument('--results_path', type=str, default='', help='')

    args = parser.parse_args()

args.resume = str2bool(args.resume)
args.disable_cuda = str2bool(args.disable_cuda)
args.with_baseline = str2bool(args.with_baseline)

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

input_size = [20]
args.n_outputs_enc = input_size[0] // 2
n_hidden = 0
tau_d = 5
tau_r = tau_d
args.dt = args.T
args.results_path = make_results_dir('predictive_softmax_', args)

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
                                  Nhid_mlp=[],
                                  num_mlp_layers=0,
                                  num_conv_layers=0,
                                  device=args.device
                                  ).to(args.device)

    encoder_optimizer = SNNSGD([{'params': encoding_network.parameters(), 'ls': True, 'baseline': False}],
                               lr=args.lr)

    if args.decoding == 'rate':
        n_inputs_decoder = args.n_outputs_enc
    else:
        n_inputs_decoder = args.n_outputs_enc * tau_d

    possible_outputs = get_possible_outputs(input_size[0])
    n_possible_outputs = len(possible_outputs)

    decoding_network, optimizer = get_decoder(args.decoder_type, args.device, args.lr,
                                              in_features=n_inputs_decoder,
                                              hid_features=(args.n_outputs_enc * tau_d)//2,
                                              out_features=n_possible_outputs,
                                              sigmoid=False,
                                              softmax=True
                                              )
    loss_fn = torch.nn.NLLLoss()
    acc_best = 0

    for ite in range(args.n_examples_train):

        if (ite + 1) % args.test_period == 0:
            print('Ite: %d' % (ite + 1))
            acc, enc_out, x_test = get_acc_predictive_softmax(encoding_network, decoding_network, tau_d, args.delta, possible_outputs, args)

            if acc > acc_best:
                acc_best = acc
                accs[i] = acc

                np.save(args.results_path + '/outputs_enc_trial_%d.npy' % i, enc_out.numpy())
                np.save(args.results_path + '/accs.npy', accs)
                np.save(args.results_path + '/xtest_trial_%d.npy' % i, x_test.numpy())
                torch.save(encoding_network.state_dict(), args.results_path + '/encoding_network_trial_%d.pt' % i)
                torch.save(decoding_network.state_dict(), args.results_path + '/decoding_network_trial_%d.pt' % i)

        if args.delta <= 0:
            blob_1 = make_blob(input_size[0], args.T)
            blob_2 = make_blob(input_size[0], args.T)
        else:
            blob_1 = make_blob(input_size[0], args.T + args.delta)
            blob_2 = make_blob(input_size[0], args.T + args.delta)

        inputs = (blob_1 + blob_2).unsqueeze(0)
        inputs[inputs == 2.] = 1

        if args.delta > 0:
            targets = torch.cat((torch.zeros([1, input_size[0], tau_d]), inputs[:, :, (tau_d + args.delta):]), dim=-1)
        elif args.delta < 0:
            targets = torch.cat((torch.zeros([1, input_size[0], tau_d]), inputs[:, :, (tau_d + args.delta): args.T + args.delta]), dim=-1)
        else:
            targets = inputs

        decoding_network.train()
        encoding_network.train()

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        encoder_outputs = torch.Tensor().to(args.device)
        enc_loss = 0
        dec_loss = 0

        with torch.no_grad():
            length = encoding_network.out_layer.out_layer.memory_length + 1
            refractory_sig = torch.zeros([input_size[0], length])
            for t in range(length):
                encoding_network(refractory_sig[:, t].to(args.device))

        for t in range(tau_d):
            enc_outputs, enc_probas = encoding_network(inputs[:, :, t].to(args.device))
            encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, tau_d)

        for t in range(tau_d, args.T):
            enc_outputs, enc_probas = encoding_network(inputs[:, :, t].to(args.device))

            encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, tau_d)

            enc_loss = torch.mean(enc_outputs * torch.log(1e-7 + enc_probas / args.r)
                                      + (1 - enc_outputs) * torch.log(1e-7 + (1 - enc_probas) / (1 - args.r)))
            enc_loss.backward()

            decoder_outputs = get_decoder_outputs(decoding_network, encoder_outputs, args.decoding)

            target_one_hot_idx = get_one_hot_index(targets[0, :, t], possible_outputs)
            dec_loss = loss_fn(decoder_outputs, target_one_hot_idx.to(args.device))
            dec_loss.backward()

            encoder_optimizer.step(dec_loss.detach() + args.beta * enc_loss)
            encoder_optimizer.zero_grad()

            optimizer.step()
            optimizer.zero_grad()
