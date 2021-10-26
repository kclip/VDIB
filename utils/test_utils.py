import numpy as np
import torch
import os
from snn.utils.utils_snn import refractory_period

from utils.misc import get_encoder_outputs, get_encoded_image, get_decoder_outputs, get_decoder
from models.lenet import LeNet5
from utils.data_utils import make_blob, get_possible_outputs, get_one_hot_index
from models.encoders import DFAEncoder


def get_lenet_acc(images, true_labels, device):
    net = LeNet5().eval()
    net.load_state_dict(torch.load(os.getcwd() + r'/weights/lenet_epoch=12_test_acc=0.991.pth'))
    net = net.to(device)

    imgs_resized = torch.zeros([len(images), 1, 32, 32]).to(device)
    imgs_resized[:, :, 2:-2, 2:-2] = images.reshape([-1, 28, 28]).unsqueeze(1)

    preds = net(imgs_resized).cpu().detach().numpy()
    class_preds = np.argmax(preds, axis=1)

    return np.sum(class_preds == true_labels) / len(imgs_resized)


def save_sigs(args, outputs, enc_out, enc_hidden, targets, true_labels, encoding_network=None, decoding_network=None, trial=0):
    np.save(args.results_path + '/img_recon_trial_%d.npy' % trial, outputs.numpy())
    np.save(args.results_path + '/enc_outputs_trial_%d.npy' % trial, enc_out.numpy())
    np.save(args.results_path + '/img_trial_%d.npy' % trial, targets.numpy())
    if true_labels is not None:
        np.save(args.results_path + '/labels_trial_%d.npy' % trial, true_labels)
    if enc_hidden is not None:
        np.save(args.results_path + '/enc_hidden_trial_%d.npy' % trial, enc_hidden.numpy())
    if encoding_network is not None:
        torch.save(encoding_network.state_dict(), args.results_path + '/encoding_network_trial_%d.pt' % trial)
    if decoding_network is not None:
        torch.save(decoding_network.state_dict(), args.results_path + '/decoding_network_trial_%d.pt' % trial)


def final_acc_predictive(path, T=10000, delta=0, decoder_type='linear', decoding='', lr=0, device='cpu'):
    tau_d = 5
    possible_outputs = torch.Tensor(get_possible_outputs(20))

    encoding_network = DFAEncoder([20],
                                  10,
                                  mode='DFA',
                                  Nhid_conv=[],
                                  Nhid_mlp=[],
                                  num_mlp_layers=0,
                                  num_conv_layers=0,
                                  device=device
                                  ).to(device)

    if decoding == 'rate':
        n_inputs_decoder = 10
    else:
        n_inputs_decoder = 10 * tau_d

    decoding_network, optimizer = get_decoder(decoder_type, device, lr,
                                              T=T,
                                              in_features=n_inputs_decoder,
                                              hid_features=(10 * tau_d) // 2,
                                              out_features=len(possible_outputs),
                                              sigmoid=False,
                                              softmax=True
                                              )
    print(decoding_network)
    accs = []

    for i in range(5):
        try:
            encoding_network.load_state_dict(torch.load(path + r'\encoding_network_trial_%d.pt' % i, map_location=device))
            decoding_network.load_state_dict(torch.load(path + r'\decoding_network_trial_%d.pt' % i, map_location=device))

            encoding_network.eval()
            decoding_network.eval()

            if delta <= 0:
                blob_1 = make_blob(encoding_network.input_shape[0], T)
                blob_2 = make_blob(encoding_network.input_shape[0], T)
            else:
                blob_1 = make_blob(encoding_network.input_shape[0], T + delta)
                blob_2 = make_blob(encoding_network.input_shape[0], T + delta)

            inputs = (blob_1 + blob_2).unsqueeze(0)
            inputs[inputs == 2.] = 1

            if delta > 0:
                targets_test = torch.cat((torch.zeros([1, encoding_network.input_shape[0], tau_d]), inputs[:, :, (tau_d + delta):]), dim=-1)
            elif delta < 0:
                targets_test = torch.cat((torch.zeros([1, encoding_network.input_shape[0], tau_d]), inputs[:, :, (tau_d + delta): T + delta]), dim=-1)
            else:
                targets_test = inputs

            enc_out = torch.zeros([T, encoding_network.out_layer.n_output_neurons])
            enc_out_potential = torch.zeros([T, encoding_network.out_layer.n_output_neurons])
            outputs_spiking = torch.Tensor()
            targets = torch.LongTensor()

            with torch.no_grad():
                encoder_outputs = torch.Tensor().to(device)
                refractory_period(encoding_network)

                length = encoding_network.out_layer.out_layer.memory_length + 1
                refractory_sig = torch.zeros([1, 20, length])

                for t in range(length):
                    encoding_network(refractory_sig[:, :, t].to(device))

                for t in range(tau_d):
                    enc_outputs, enc_probas = encoding_network(inputs[:, :, t].to(device))
                    encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, tau_d)

                for t in range(tau_d, T):
                    enc_outputs, enc_probas = encoding_network(inputs[:, :, t].to(device))
                    encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, tau_d)
                    enc_out[t] = enc_outputs
                    enc_out_potential[t] = encoding_network.out_layer.out_layer.potential

                    targets = torch.cat((targets, targets_test[:, :, t]), dim=0)
                    decoder_output = get_decoder_outputs(decoding_network, encoder_outputs, decoding).argmax(-1).cpu()
                    outputs_spiking = torch.cat((outputs_spiking, possible_outputs[decoder_output]))

            acc = torch.nn.functional.mse_loss(outputs_spiking, targets)
            print(acc)

            accs.append(acc)
            np.save(path + '/accs_final.npy', np.array(accs))
            np.save(path + '/enc_out_final_trial_%d.npy' % i, enc_out.numpy())
            np.save(path + '/outputs_spiking_trial_%d.npy' % i, outputs_spiking.numpy())
            np.save(path + '/inputs_spiking_trial_%d.npy' % i, inputs.numpy())

        except FileNotFoundError:
            break


def final_acc_reconstruction(path, T=30, n_outputs_enc=256, decoder_type='conv', encoding='time', decoding='', lr=0, device='cpu', n_examples_test=None):
    from torchvision import datasets, transforms

    size = 28
    input_size = [size ** 2]

    hidden_neurons = [600]
    tau_d = T
    digits = [i for i in range(10)]

    transform = transforms.Compose([transforms.Resize([size, size]), transforms.ToTensor()])

    mnist_test = datasets.MNIST('../data', train=False,
                                transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, shuffle=True, batch_size=1)

    encoding_network = DFAEncoder(input_size,
                                  n_outputs_enc,
                                  mode='DFA',
                                  Nhid_conv=[],
                                  Nhid_mlp=hidden_neurons,
                                  num_mlp_layers=len(hidden_neurons),
                                  num_conv_layers=0,
                                  device=device
                                  ).to(device)

    if decoding == 'rate':
        n_inputs_decoder = n_outputs_enc
    else:
        n_inputs_decoder = n_outputs_enc * tau_d

    decoding_network, optimizer = get_decoder(decoder_type, device, lr,
                                              T=T,
                                              in_features=n_inputs_decoder,
                                              hid_features=(n_outputs_enc * tau_d) // 2,
                                              out_features=input_size[0],
                                              )

    accs = []
    for j in range(5):
        try:
            encoding_network.load_state_dict(torch.load(path + r'/encoding_network_trial_%d.pt' % j, map_location=device))
            decoding_network.load_state_dict(torch.load(path + r'/decoding_network_trial_%d.pt' % j, map_location=device))

            encoding_network.eval()
            decoding_network.eval()

            true_labels = np.array([])

            test_iterator = iter(test_loader)

            if n_examples_test is None:
                n_examples_test = len(test_iterator)

            outputs = torch.zeros([n_examples_test, size * size])

            for i in range(n_examples_test):
                x_test, target, label = get_encoded_image(test_iterator, test_loader, digits, encoding, T)
                true_labels = np.hstack((true_labels, label.numpy()))

                with torch.no_grad():
                    encoder_outputs = torch.Tensor().to(device)
                    refractory_period(encoding_network)

                    length = encoding_network.out_layer.out_layer.memory_length + 1
                    refractory_sig = torch.zeros([1, 28 ** 2, length])

                    for t in range(length):
                        encoding_network(refractory_sig[:, :, t].to(device))

                    x_test = x_test.unsqueeze(0)
                    for t in range(T):
                        enc_outputs, enc_probas = encoding_network(x_test[:, :, t].to(device))
                        encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, T)

                    outputs[i] = get_decoder_outputs(decoding_network, encoder_outputs, decoding)

            acc = get_lenet_acc(outputs, true_labels, device)
            print(acc)
            accs.append(acc)

            np.save(path + '/acc_final.npy', np.array(accs))

        except FileNotFoundError:
            print('File not found at ite %d' % j)
            print('Path:', path)
            print(os.listdir(path))

            break


def final_acc_reconstruction_mnistdvs(path, dataset_path, device, dt, decoding, decoder_type, n_examples_test):
    import tables
    from neurodata.load_data import create_dataloader

    dataset = tables.open_file(dataset_path)
    input_size = [2, 26, 26]
    dataset.close()

    sample_length = 2000000
    T = int(sample_length / dt)
    tau_d = T

    digits = [i for i in range(10)]

    n_outputs_enc = 256

    # Create dataloaders
    _, test_dl = create_dataloader(dataset_path, batch_size=1, size=input_size, classes=digits, sample_length_train=sample_length,
                                   sample_length_test=sample_length, dt=dt, polarity=True, ds=1, shuffle_test=True, num_workers=0)

    hidden_neurons = [800]
    encoding_network = DFAEncoder([np.prod(input_size)],
                                  n_outputs_enc,
                                  Nhid_conv=[],
                                  Nhid_mlp=hidden_neurons,
                                  num_mlp_layers=len(hidden_neurons),
                                  num_conv_layers=0,
                                  device=device
                                  ).to(device)


    if decoding == 'rate':
        n_inputs_decoder = n_outputs_enc
    else:
        n_inputs_decoder = n_outputs_enc * tau_d

    decoding_network, optimizer = get_decoder(decoder_type, device, 0,
                                              T=T,
                                              in_features=n_inputs_decoder,
                                              hid_features=(n_outputs_enc * tau_d)//2,
                                              out_features=28 * 28,
                                              )


    accs = []
    for j in range(5):
        try:
            encoding_network.load_state_dict(torch.load(path + r'/encoding_network_trial_%d.pt' % j, map_location=device))
            decoding_network.load_state_dict(torch.load(path + r'/decoding_network_trial_%d.pt' % j, map_location=device))

            encoding_network.eval()
            decoding_network.eval()

            test_iterator = iter(test_dl)

            if n_examples_test is None:
                n_examples_test = len(test_iterator)

            outputs = torch.zeros([n_examples_test, 28 * 28])

            enc_out = torch.zeros([n_examples_test, encoding_network.out_layer.n_output_neurons, T])

            targets = torch.Tensor()
            true_labels = np.array([])

            for i in range(n_examples_test):
                x_test, label = next(test_iterator)

                target = torch.sum(x_test, dim=(1, 2))
                targets = torch.cat((targets, target))

                label = torch.sum(label, dim=-1).argmax(-1)

                true_labels = np.hstack((true_labels, label.numpy()))

                with torch.no_grad():
                    encoder_outputs = torch.Tensor().to(device)
                    length = encoding_network.out_layer.out_layer.memory_length + 1
                    refractory_sig = torch.zeros([1, length, 2, 26, 26])
                    for t in range(length):
                        encoding_network(refractory_sig[:, t].to(device))

                    for t in range(T):
                        enc_outputs, enc_probas = encoding_network(x_test[:, t].to(device))
                        encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, T)
                        enc_out[i, :, t] = enc_outputs

                    outputs[i] = get_decoder_outputs(decoding_network, encoder_outputs, decoding)

            acc = get_lenet_acc(outputs, true_labels, device)
            print(acc)
            accs.append(acc)

            np.save(path + '/acc_final.npy', np.array(accs))
            np.save(path + '/enc_outs_final_trial_%d.npy' % j, enc_out.numpy())
            np.save(path + '/true_labels_final_%d.npy' % j, true_labels)


        except FileNotFoundError:
            print('File not found at ite %d' % j)
            print('Path:', path)
            print(os.listdir(path))

            break


def get_acc_predictive_softmax(encoding_network, decoding_network, tau_d, delta, possible_outputs, args):
    encoding_network.eval()
    decoding_network.eval()

    decoder_outputs = torch.LongTensor()
    enc_out = torch.zeros([encoding_network.output_shape, args.T_test])

    if args.delta <= 0:
        blob_1 = make_blob(encoding_network.input_shape[0], args.T_test)
        blob_2 = make_blob(encoding_network.input_shape[0], args.T_test)
    else:
        blob_1 = make_blob(encoding_network.input_shape[0], args.T_test + args.delta)
        blob_2 = make_blob(encoding_network.input_shape[0], args.T_test + args.delta)


    x_test = (blob_1 + blob_2).unsqueeze(0)
    x_test[x_test == 2.] = 1

    if delta > 0:
        targets_test = torch.cat((torch.zeros([1, encoding_network.input_shape[0], tau_d]), x_test[:, :, (tau_d + delta):]), dim=-1)
    elif delta < 0:
        targets_test = torch.cat((torch.zeros([1, encoding_network.input_shape[0], tau_d]), x_test[:, :, (tau_d + delta): args.T_test + delta]), dim=-1)
    else:
        targets_test = x_test

    targets = torch.LongTensor()

    with torch.no_grad():
        encoder_outputs = torch.Tensor().to(args.device)

        length = encoding_network.out_layer.out_layer.memory_length + 1
        refractory_sig = torch.zeros([1, encoding_network.input_shape[0], length])
        for t in range(length):
            encoding_network(refractory_sig[:, :, t].to(args.device))

        for t in range(tau_d):
            enc_outputs, enc_probas = encoding_network(x_test[:, :, t].to(args.device))
            encoder_outputs = get_encoder_outputs(enc_outputs.detach(), decoding_network, encoder_outputs, tau_d)

        for t in range(tau_d, args.T_test - max(0, delta)):
            enc_outputs, enc_probas = encoding_network(x_test[:, :, t].to(args.device))
            encoder_outputs = get_encoder_outputs(enc_outputs.detach(), decoding_network, encoder_outputs, tau_d)

            enc_out[:, t] = enc_outputs.cpu()

            target_one_hot_idx = get_one_hot_index(targets_test[0, :, t], possible_outputs)

            targets = torch.cat((targets, target_one_hot_idx), dim=0)

            decoder_output = get_decoder_outputs(decoding_network, encoder_outputs, args.decoding)

            decoder_outputs = torch.cat((decoder_outputs, decoder_output.argmax(-1).cpu()))

        acc = float(torch.sum(decoder_outputs == targets, dtype=torch.float)) / decoder_outputs.numel()

    return acc, enc_out, x_test


def get_acc_reconstruction(encoding_network, decoding_network, test_loader, args, digits, size, acc_best, trial=0):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoding_network.eval()
    decoding_network.eval()

    loss_fn = torch.nn.BCELoss()
    loss = 0
    targets = torch.Tensor()
    true_labels = np.array([])

    test_iter = iter(test_loader)
    if args.n_examples_test is None:
        args.n_examples_test = len(test_iter)

    outputs = torch.zeros([args.n_examples_test, size*size])
    enc_out = torch.zeros([args.n_examples_test, encoding_network.out_layer.n_output_neurons, args.T])
    enc_hidden = torch.Tensor()

    for i in range(args.n_examples_test):
        x_test, target, label = get_encoded_image(test_iter, test_loader, digits, args.encoding, args.T)

        true_labels = np.hstack((true_labels, label.numpy()))
        targets = torch.cat((targets, target.unsqueeze(0)), dim=0)

        with torch.no_grad():
            encoder_outputs = torch.Tensor().to(args.device)
            refractory_period(encoding_network)

            length = encoding_network.out_layer.out_layer.memory_length + 1
            refractory_sig = torch.zeros([1, 28**2, length])
            for t in range(length):
                encoding_network(refractory_sig[:, :, t].to(args.device))

            x_test = x_test.unsqueeze(0)
            for t in range(args.T):
                enc_outputs, enc_probas = encoding_network(x_test[:, :, t].to(args.device))
                encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, args.T)
                enc_out[i, :, t] = enc_outputs
                enc_hidden = torch.cat((enc_hidden, encoding_network.hidden_hist[:, -1].cpu()))

            outputs[i] = get_decoder_outputs(decoding_network, encoder_outputs, args.decoding)
            loss += loss_fn(outputs[i], target[0])

    acc = get_lenet_acc(outputs, true_labels, args.device)
    if acc > acc_best:
        save_sigs(args, outputs, enc_out, enc_hidden, targets, true_labels, encoding_network, decoding_network, trial)

    return acc


def get_acc_reconstruction_mnistdvs(encoding_network, decoding_network, test_iter, args, T, acc_best, trial=0):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoding_network.eval()
    decoding_network.eval()

    outputs = torch.zeros([args.n_examples_test, 28 * 28])
    enc_out = torch.zeros([args.n_examples_test, encoding_network.out_layer.n_output_neurons, T])
    enc_hidden = torch.Tensor()

    targets = torch.Tensor()
    true_labels = np.array([])

    for i in range(args.n_examples_test):
        x_test, label = next(test_iter)

        target = torch.sum(x_test, dim=(1, 2))
        targets = torch.cat((targets, target))

        label = torch.sum(label, dim=-1).argmax(-1)

        true_labels = np.hstack((true_labels, label.numpy()))

        with torch.no_grad():
            encoder_outputs = torch.Tensor().to(args.device)
            length = encoding_network.out_layer.out_layer.memory_length + 1
            refractory_sig = torch.zeros([1, length, 2, 26, 26])
            for t in range(length):
                encoding_network(refractory_sig[:, t].to(args.device))

            for t in range(T):
                enc_outputs, enc_probas = encoding_network(x_test[:, t].to(args.device))
                encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, T)
                enc_out[i, :, t] = enc_outputs
                enc_hidden = torch.cat((enc_hidden, encoding_network.hidden_hist[:, -1].cpu()))

            outputs[i] = get_decoder_outputs(decoding_network, encoder_outputs, args.decoding)


    acc = get_lenet_acc(outputs, true_labels, args.device)
    if acc > acc_best:
        save_sigs(args, outputs, enc_out, enc_hidden, targets, true_labels, encoding_network, decoding_network, trial)

    return acc


def get_acc_classification_mnistdvs(encoding_network, decoding_network, test_iter, args, T, acc_best, trial):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoding_network.eval()
    decoding_network.eval()

    predictions = torch.zeros([args.n_examples_test])
    enc_out = torch.zeros([args.n_examples_test, encoding_network.out_layer.n_output_neurons, T])
    enc_hidden = torch.Tensor()

    true_labels = np.array([])

    for i in range(args.n_examples_test):
        x_test, label = next(test_iter)

        label = torch.sum(label, dim=-1).argmax(-1)

        true_labels = np.hstack((true_labels, label.numpy()))

        with torch.no_grad():
            encoder_outputs = torch.Tensor().to(args.device)
            length = encoding_network.out_layer.out_layer.memory_length + 1
            refractory_sig = torch.zeros([1, length, 2, 26, 26])
            for t in range(length):
                encoding_network(refractory_sig[:, t].to(args.device))

            for t in range(T):
                enc_outputs, enc_probas = encoding_network(x_test[:, t].to(args.device))
                encoder_outputs = get_encoder_outputs(enc_outputs, decoding_network, encoder_outputs, T)
                enc_out[i, :, t] = enc_outputs
                enc_hidden = torch.cat((enc_hidden, encoding_network.hidden_hist[:, -1].cpu()))

            predictions[i] = get_decoder_outputs(decoding_network, encoder_outputs, args.decoding).argmax(-1)


    acc = torch.sum(predictions == torch.Tensor(true_labels), dtype=torch.float) / len(predictions)
    if acc > acc_best:
        save_sigs(args, predictions, enc_out, enc_hidden, torch.Tensor(true_labels), true_labels, encoding_network, decoding_network, trial)

    return acc
