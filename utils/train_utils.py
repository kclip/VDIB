from snn.models.SNN import BinarySNN, LayeredSNN


def update_snn_decoder(decoding_network, args, eligibility_trace_dec=None):
    for parameter in decoding_network.get_gradients():
        eligibility_trace_dec[parameter] = args.kappa * eligibility_trace_dec[parameter] \
                                          + (1 - args.kappa) * (-decoding_network.get_gradients()[parameter])
        decoding_network.get_parameters()[parameter] -= args.lr * eligibility_trace_dec[parameter]


def update_ann_decoder(loss, optimizer):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def update_snn_encoder(encoding_network, baseline_num, baseline_den, eligibility_trace_enc, learning_signal, enc_loss, dec_loss, lr, beta, kappa):
    if learning_signal is not None:
        learning_signal.mul_(kappa).add_(dec_loss.detach() + beta * enc_loss, alpha=1 - kappa)
    else:
        learning_signal = dec_loss.detach() + beta * enc_loss

    for parameter in encoding_network.get_gradients():
        if baseline_num is not None:
            baseline_num[parameter].mul_(kappa).add_(eligibility_trace_enc[parameter].pow(2).mul(learning_signal), alpha=1 - kappa)
            baseline_den[parameter].mul_(kappa).add_(eligibility_trace_enc[parameter].pow(2), alpha=1 - kappa)
            baseline = baseline_num[parameter] / (baseline_den[parameter] + 1e-07)

            update_enc = (learning_signal - baseline) * eligibility_trace_enc[parameter]
        else:
            update_enc = learning_signal * eligibility_trace_enc[parameter]

        encoding_network.get_parameters()[parameter] -= lr * update_enc


def update_layered_encoder(optimizer, enc_loss, dec_loss, beta):
    optimizer.step(dec_loss.detach() + beta * enc_loss.detach())
    optimizer.zero_grad()


def update_system(enc_loss, dec_loss, beta, encoder_optimizer=None, optimizer=None):
    update_layered_encoder(encoder_optimizer, enc_loss, dec_loss, beta)
    update_ann_decoder(dec_loss, optimizer)

