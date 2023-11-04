import torch
from stream.gen import RNG

class FSUAdd(torch.nn.Module):
    """
    This module is for unary addition for arbitrary scale, including scaled/non-scaled, unipolar/bipolar.
    """
    def __init__(self, 
                    mode="bipolar", 
                    scaled=True, 
                    scale=None, 
                    dim=0, 
                    depth=10, 
                    entry=None, 
                    stype=torch.float):
        super(FSUAdd, self).__init__()
        
        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # scale is an arbitrary value that larger than 0
        self.scale = scale
        # dimension to do reduced sum
        self.dim = dim
        # max value in the accumulator
        self.acc_max = 2**(depth-2)
        # min value in the accumulator
        self.acc_min = -2**(depth-2)
        # number of entries in dim to do reduced sum
        self.entry = entry
        self.stype = stype
        
        # the carry scale at the output
        self.scale_carry = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # accumulator for (PC - offset)
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.first = True

    def forward(self, input, scale=None, entry=None):
        if self.first:
            if self.scaled is True:
                if self.scale is None:
                    self.scale_carry.fill_(input.size()[self.dim])
                else:
                    self.scale_carry.fill_(self.scale)
                if scale is not None:
                    # runtime scale will override the default value
                    self.scale_carry.fill_(scale)
            if self.mode == "bipolar":
                if self.entry is None:
                    self.offset.data = (input.size()[self.dim] - self.scale_carry)/2
                else:
                    self.offset.data = (self.entry - self.scale_carry)/2
                if entry is not None:
                    # runtime entry will override the default value in bipolar mode
                    self.offset.data = (entry - self.scale_carry)/2
            self.first = False
        acc_delta = torch.sum(input.type(torch.float), self.dim) - self.offset
        self.accumulator.data = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
        output = torch.gt(self.accumulator, self.scale_carry).type(torch.float)
        self.accumulator.sub_(output * self.scale_carry).clamp_(self.acc_min, self.acc_max)
        return output.type(self.stype)


class FSUAdduGEMM(torch.nn.Module):
    """
    This module is for unary addition in uGEMM, including scaled/non-scaled, unipolar/bipolar.
    """
    def __init__(self, 
                 mode="bipolar", 
                 scaled=True, 
                 dim=0, 
                 stype=torch.float):
        super(FSUAdduGEMM, self).__init__()
        
        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # dimension to do reduce sum
        self.dim = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.dim.fill_(dim)
        self.stype = stype
        
        # upper bound for accumulation counter in scaled mode
        # it is the number of inputs, e.g., the size along the dim dimension
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input):
        self.acc_bound.fill_(input.size()[self.dim.item()])
        if self.mode == "bipolar":
            self.offset.fill_((self.acc_bound.item()-1)/2)
        self.accumulator.data = self.accumulator.add(torch.sum(input.type(torch.float), self.dim.item()))
        
        if self.scaled is True:
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(self.stype)


class GainesAdd(torch.nn.Module):
    """
    this module is for Gaines addition.
    1) MUX for scaled addition
    2) OR gate for non-scaled addition
    """
    def __init__(self, 
                 mode="bipolar", 
                 scaled=True, 
                 dim=0, 
                 rng="Sobol", 
                 rng_dim=5, 
                 rng_width=8, 
                 rtype=torch.float, 
                 stype=torch.float):
        super(GainesAdd, self).__init__()
        
        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        if self.mode == "bipolar" and self.scaled is False:
            raise ValueError("Non-scaled addition for biploar data is not supported in Gaines approach.")
        # dimension to do reduce sum
        self.dim = dim
        self.stype = stype
        self.rng = RNG(rng_width, rng_dim, rng, rtype=rtype)()
        self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        
    def forward(self, input):
        if self.scaled is True:
            randNum = self.rng[self.rng_idx.item()]
            assert randNum.item() < input.size()[self.dim], "randNum should be smaller than the dimension size of addition."
            # using a MUX for both unipolar and bipolar
            output = torch.unbind(torch.index_select(input, self.dim, randNum.type(torch.long).view(1)), self.dim)[0]
            self.rng_idx.data = self.rng_idx.add(1)%self.rng.numel()
        else:
            # only support unipolar data using an OR gate
            output = torch.gt(torch.sum(input, self.dim), 0)
            
        return output.type(self.stype)
    