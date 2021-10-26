import torch
import numpy as np

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >=0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0


        return grad_input


class SigmoidStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).type(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        res = torch.sigmoid(input)
        return res*(1-res)*grad_output



class SurrGradSpike(torch.autograd.Function):
    """
    From https://github.com/fzenke/spytorch
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad



class HookFunction(torch.autograd.Function):
    """"
    From https://github.com/ChFrenkel/DirectRandomTargetProjection (code under Apache v2.0 license)
    """
    @staticmethod
    def forward(ctx, input, e, fixed_fb_weights, mode):
        ctx.save_for_backward(input, e, fixed_fb_weights)
        ctx.in1 = mode
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, e, fixed_fb_weights = ctx.saved_variables
        mode = ctx.in1
        if mode == 'DFA':
            grad_output_est = e.mm(fixed_fb_weights.view(-1, np.prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        else:
            grad_output_est = grad_output

        return grad_output_est, None, None, None


trainingHook = HookFunction.apply

smooth_step = SmoothStep().apply
smooth_sigmoid = SigmoidStep().apply
surrgrad = SurrGradSpike().apply
