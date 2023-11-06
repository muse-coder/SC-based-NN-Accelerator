import torch
import math
from kernel.utils import conv2d_output_shape, num2tuple
from torch.cuda.amp import autocast
from SC_GEMM import *
class SC_Conv2d(torch.nn.Conv2d):
    """
    This module is the 2d conv layer, with binary input and binary output
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 binary_weight=None,
                 binary_bias=None,
                 bitwidth=8,
                 keep_res="input",  # keep the resolution of input/output
                 more_res="input",  # assign more resolution to input/weight
                 rounding="round"):
        super(SC_Conv2d, self).__init__(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        groups,
                                        bias,
                                        padding_mode)

        assert groups == 1, "Supported group number is 1."
        assert padding_mode == 'zeros', "Supported padding_mode number is 'zeros'."

        # weight and bias
        if binary_weight is not None:
            self.weight.data = binary_weight

        if bias and (binary_bias is not None):
            self.bias.data = binary_bias

        # bitwidth of abs
        if isinstance(bitwidth, tuple):
            self.bw_input, self.bw_wght = (bitwidth[0] - 1, bitwidth[1] - 1)
        else:
            if keep_res == "input":
                self.bw_input, self.bw_wght = (bitwidth - 1, bitwidth - 1)
            elif keep_res == "output":
                if bitwidth % 2 == 0:
                    self.bw_input, self.bw_wght = (int(bitwidth / 2 - 1), int(bitwidth / 2 - 1))
                else:
                    if more_res == "input":
                        self.bw_input, self.bw_wght = (int((bitwidth + 1) / 2 - 1), int((bitwidth - 1) / 2 - 1))
                    elif more_res == "weight":
                        self.bw_input, self.bw_wght = (int((bitwidth - 1) / 2 - 1), int((bitwidth + 1) / 2 - 1))
                    else:
                        raise ValueError(
                            "more_res should be either 'input' or 'weight' when bitwidth is not a tuple and keep_res is 'output'.")
            else:
                raise ValueError("keep_res should be either 'input' or 'output' when bitwidth is not a tuple.")

        # max abs value
        self.max_abs_input = 2 ** self.bw_input
        self.max_abs_wght = 2 ** self.bw_wght

        # rounding mode
        self.rounding = rounding

        self.rshift_input = None
        self.rshift_wght = None
        self.rshift_output = None

    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        with torch.no_grad():
            if self.rshift_input is None:
                input_max_int = input.abs().max().log2()
                if self.rounding == "round":
                    input_max_int = input_max_int.round()
                elif self.rounding == "floor":
                    input_max_int = input_max_int.floor()
                elif self.rounding == "ceil":
                    input_max_int = input_max_int.ceil()
                self.rshift_input = input_max_int - self.bw_input

            if self.rshift_wght is None:
                wght_max_int = self.weight.abs().max().log2()
                if self.rounding == "round":
                    wght_max_int = wght_max_int.round()
                elif self.rounding == "floor":
                    wght_max_int = wght_max_int.floor()
                elif self.rounding == "ceil":
                    wght_max_int = wght_max_int.ceil()
                self.rshift_wght = wght_max_int - self.bw_wght

            if self.rshift_output is None:
                self.rshift_output = 0 - self.rshift_input - self.rshift_wght

            # all data are in NCHW
            output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size,
                                              dilation=self.dilation, pad=self.padding, stride=self.stride)

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size()[0], -1)
        mm_out = SCLinearFunction.apply(input_reshape, weight, None, self.rshift_input, self.rshift_wght,
                                         self.rshift_output, self.max_abs_input, self.max_abs_wght)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])


# Inherit from Function
class SCLinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None,
                rshift_input=3,
                rshift_wght=3,
                rshift_output=3,
                max_abs_input=128,
                max_abs_wght=128):
        ctx.save_for_backward(input, weight, bias)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # input preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # round input to (bot, top)
        bot_input = 0 - max_abs_input
        top_input = max_abs_input - 1
        input_round = torch.empty(0, device=input.device)
        if (rshift_input < 0):
            rshift_input = -rshift_input
            # torch.round(input << rshift_input, out=input_round)
            torch.round(input * (2 ** rshift_input), out=input_round)
        else:
            print("rshift >=0 ")
        torch.clamp(input_round.unsqueeze_(1), bot_input, top_input, out=input_round)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # weight preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # round input to (bot, top)
        bot_wght = 0 - max_abs_wght
        top_wght = max_abs_wght - 1
        wght_round = torch.empty(0, device=input.device)
        if (rshift_wght < 0):
            torch.round(weight * (2 ** -rshift_wght), out=wght_round)
        torch.clamp(wght_round.unsqueeze_(0), bot_wght, top_wght, out=wght_round)
        # input_round_reshape =
        output = torch.empty(0, device=weight.device)
        torch.matmul(input_round, wght_round.transpose(1, 2), out=output)

        new_wght_round = wght_round.transpose(1, 2)
        new_rshift_output = int(abs(rshift_output))
        # output = (output >> int(abs(rshift_output))).squeeze_(1)
        output = (output / (2 ** new_rshift_output)).squeeze_(1)

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None