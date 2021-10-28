import torch


class EpropLoss(object):
    """"
    Computes per-layer 'pseudo losses' to compute local gradients + out_layer loss
    """
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, s, target):
        loss = []
        for i in range(len(s) - 1):
            loss.append(torch.mean(s[i]))
        loss.append(-self.loss_fn(s[-1], target))

        return sum(loss)
