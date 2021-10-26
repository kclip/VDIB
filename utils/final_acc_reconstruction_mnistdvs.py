import argparse
import torch
from snn.utils.misc import str2bool

from utils.test_utils import final_acc_reconstruction_mnistdvs

""""
Testing for a system trained on the MNIST-DVS reconstruction task
"""

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--path', default=r"/home")
    parser.add_argument('--dataset_path', default=r"/home/datasets/mnist-dvs/mnist_dvs_events.hdf5")

    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')
    parser.add_argument('--decoder_type', type=str, default='conv', help='')
    parser.add_argument('--decoding', type=str, default='', help='')
    parser.add_argument('--dt', type=int, default=25000, help='')

    args = parser.parse_args()

args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

final_acc_reconstruction_mnistdvs(args.path, args.dataset_path, args.device, args.dt, args.decoding, args.decoder_type, None)
