import argparse
import torch
from snn.utils.misc import str2bool

from utils.test_utils import final_acc_predictive

""""
Testing for a system trained on the prediction task
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

final_acc_predictive(args.path, T=1000, delta=-1, decoder_type='linear', decoding='', lr=0, device='cpu')
