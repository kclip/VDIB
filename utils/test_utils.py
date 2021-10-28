import numpy as np
import torch
from models.lenet import LeNet5
import os
import tqdm


def get_lenet_acc(images, true_labels, device):
    net = LeNet5().eval()
    net.load_state_dict(torch.load(os.getcwd() + r'/weights/lenet_epoch=12_test_acc=0.991.pth'))
    net = net.to(device)

    imgs_resized = torch.zeros([len(images), 1, 32, 32]).to(device)
    imgs_resized[:, :, 2:-2, 2:-2] = images.reshape([-1, 28, 28]).unsqueeze(1)

    preds = net(imgs_resized).cpu().detach().numpy()
    class_preds = np.argmax(preds, axis=1)

    print(class_preds, true_labels, len(class_preds), len(true_labels), len(imgs_resized))

    return np.sum(class_preds == true_labels) / len(imgs_resized)


def get_acc_reconstruction_mnistdvs(encoding_network, decoding_network, test_iter, args, T, acc_best, tau_d, batch_size, digits):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoding_network.eval()
    decoding_network.eval()

    outputs = torch.Tensor().to(args.device)
    true_labels = np.array([])
    mean_rate_per_layer = [torch.Tensor() for _ in encoding_network.LIF_layers] + [torch.Tensor()]
    enc_out = torch.Tensor()

    for x_test, label in tqdm.tqdm(test_iter):
        if len(x_test) == batch_size:
            label = torch.sum(label, dim=-1).argmax(-1)
            true_labels = np.hstack((true_labels, np.array([digits[i] for i in label])))

            with torch.no_grad():
                length = encoding_network.tau_e + 1
                refractory_sig = torch.zeros_like(x_test)[:, :length]
                for t in range(length):
                    enc_outputs, _ = encoding_network(refractory_sig[:, t].to(args.device))
                encoder_outputs = enc_outputs

                for t in range(T):
                    enc_outputs, enc_probas = encoding_network(x_test[:, t].to(args.device))
                    encoder_outputs = torch.cat((encoder_outputs[:, -tau_d + 1:], enc_outputs), dim=1)

                    for i, l in enumerate(encoding_network.LIF_layers):
                        mean_rate_per_layer[i] = torch.cat((mean_rate_per_layer[i], torch.mean(l.state.S).unsqueeze(0).cpu()))
                    # mean_rate_per_layer[-1] = torch.cat((mean_rate_per_layer[-1], torch.mean(encoding_network.out_layer.spiking_history[:, :, -1]).unsqueeze(0).cpu()))

                enc_out = torch.cat((enc_out, encoder_outputs.cpu()))

                outputs = torch.cat((outputs, decoding_network(encoder_outputs.unsqueeze(1))))

    acc = get_lenet_acc(outputs, true_labels, args.device)

    if acc > acc_best:
        np.save(args.results_path + '/img_recon.npy', outputs.cpu().numpy())
        np.save(args.results_path + '/enc_outputs.npy', enc_out.cpu().numpy())
        np.save(args.results_path + '/true_labels.npy', true_labels)
        np.save(args.results_path + '/acc_best.npy', acc_best)
        torch.save(encoding_network.state_dict(), args.results_path + '/encoding_network.pt')
        torch.save(decoding_network.state_dict(), args.results_path + '/decoding_network.pt')

    return acc
