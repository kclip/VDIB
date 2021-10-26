import argparse
import torch
from snn.utils.misc import str2bool

from utils.test_utils import final_acc_reconstruction

""""
Testing for a system trained on the MNIST reconstruction task
"""

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--path', default=r"/home")
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')
    parser.add_argument('--decoder_type', type=str, default='conv', help='')
    parser.add_argument('--encoding', type=str, default='time', help='')
    parser.add_argument('--decoding', type=str, default='', help='')

    args = parser.parse_args()

args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

final_acc_reconstruction(args.path, T=30, n_outputs_enc=256, decoder_type=args.decoder_type,
                         encoding=args.encoding, decoding=args.decoding, lr=0, device=args.device, n_examples_test=None)
