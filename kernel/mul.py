import torch
from  stream.gen import RNG, SourceGen, BSGen
from  kernel.shiftreg import ShiftReg

class FSUMul(torch.nn.Module):
    """
    this module is for unary multiplication, supporting static/non-static computation, unipolar/bipolar
    input_prob_1 is a don't care if the multiplier is non-static, defualt value is None.
    if the multiplier is static, then need to input the pre-scaled input_1 to port input_prob_1 
    """
    def __init__(self,
                 bitwidth=8,
                 mode="bipolar",
                 static=False,
                 input_prob_1=None,
                 rtype=torch.float,
                 stype=torch.float):
        super(FSUMul, self).__init__()
        
        self.bitwidth = bitwidth
        self.depth = 2**bitwidth
        self.mode = mode
        self.static = static
        self.stype = stype
        self.rtype = rtype
        
        assert self.mode == "unipolar" or self.mode == "bipolar", "Unsupported mode in FSUMul."

        # the random number generator used in computation
        self.rng = RNG(
            bitwidth=self.bitwidth,
            dim=1,
            rng="Sobol",
            rtype=self.rtype)()
        
        if self.static is True:
            # the probability of input_1 used in static computation
            self.input_prob_1 = input_prob_1
            assert input_prob_1 is not None, "Static multiplier requires input_prob_1."
            # directly create an unchange bitstream generator for static computation
            self.source_gen = SourceGen(self.input_prob_1, self.bitwidth, self.mode, self.rtype)()
            self.bsg = BSGen(self.source_gen, self.rng, torch.int8)
            # rng_idx is used later as an enable signal, get update every cycled
            self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            
            # Generate two seperate bitstream generators and two enable signals for bipolar mode
            if self.mode == "bipolar":
                self.bsg_inv = BSGen(self.source_gen, self.rng, torch.int8)
                self.rng_idx_inv = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        else:
            # use a shift register to store the count of 1s in one bitstream to generate data
            self.sr = ShiftReg(depth=self.depth, stype=self.stype)
            self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            if self.mode == "bipolar":
                self.rng_idx_inv = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

    def FSUMul_forward(self, input_0, input_1=None):
        if self.static is True:
            # for input0 is 0.
            path = input_0.type(torch.int8) & self.bsg(self.rng_idx)
            # conditional update for rng index when input0 is 1. The update simulates enable signal of bs gen.
            self.rng_idx.data = self.rng_idx.add(input_0.type(torch.long))
            
            if self.mode == "unipolar":
                return path
            else:
                # for input0 is 0.
                path_inv = (1 - input_0.type(torch.int8)) & (1 - self.bsg_inv(self.rng_idx_inv))
                # conditional update for rng_idx_inv
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - input_0.type(torch.long))
                return path | path_inv
        else:
            _, source = self.sr(input_1)
            path = input_0.type(torch.int8) & torch.gt(source, self.rng[self.rng_idx]).type(torch.int8)
            self.rng_idx.data = self.rng_idx.add(input_0.type(torch.long)) % self.depth

            if self.mode == "unipolar":
                return path
            else:
                # for input0 is 0.
                path_inv = (1 - input_0.type(torch.int8)) & (1 - torch.gt(source, self.rng[self.rng_idx_inv]).type(torch.int8))
                # conditional update for rng_idx_inv
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - input_0.type(torch.long)) % self.depth
                return path | path_inv
            
    def forward(self, input_0, input_1=None):
        return self.FSUMul_forward(input_0, input_1).type(self.stype)

    
class GainesMul(torch.nn.Module):
    """
    this module is for Gaines stochastic multiplication, supporting unipolar/bipolar
    """
    def __init__(self,
                 mode="bipolar",
                 stype=torch.float):
        super(GainesMul, self).__init__()
        self.mode = mode
        self.stype = stype

    def GainesMul_forward(self, input_0, input_1):
        if self.mode == "unipolar":
            return input_0.type(torch.int8) & input_1.type(torch.int8)
        elif self.mode == "bipolar":
            return 1 - (input_0.type(torch.int8) ^ input_1.type(torch.int8))
        else:
            raise ValueError("GainesMul mode is not implemented.")
            
    def forward(self, input_0, input_1):
        return self.GainesMul_forward(input_0, input_1).type(self.stype)


class SC_Based_Mul(torch.nn.Module):
    """
    this module is for unary multiplication, supporting static/non-static computation, unipolar/bipolar
    input_prob_1 is a don't care if the multiplier is non-static, defualt value is None.
    if the multiplier is static, then need to input the pre-scaled input_1 to port input_prob_1
    """

    def __init__(self,
                 bitwidth=8,
                 mode="bipolar",
                 rng=[]):
        super(FSUMul, self).__init__()

        self.bitwidth = bitwidth
        self.depth = 2 ** bitwidth
        self.mode = mode
        self.stype = stype
        self.rtype = rtype

        assert self.mode == "unipolar" or self.mode == "bipolar", "Unsupported mode in FSUMul."

        # the random number generator used in computation
        self.rng = rng

    def FSUMul_forward(self, input_0, input_1=None):
        if self.static is True:
            # for input0 is 0.
            path = input_0.type(torch.int8) & self.bsg(self.rng_idx)
            # conditional update for rng index when input0 is 1. The update simulates enable signal of bs gen.
            self.rng_idx.data = self.rng_idx.add(input_0.type(torch.long))

            if self.mode == "unipolar":
                return path
            else:
                # for input0 is 0.
                path_inv = (1 - input_0.type(torch.int8)) & (1 - self.bsg_inv(self.rng_idx_inv))
                # conditional update for rng_idx_inv
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - input_0.type(torch.long))
                return path | path_inv
        else:
            _, source = self.sr(input_1)
            path = input_0.type(torch.int8) & torch.gt(source, self.rng[self.rng_idx]).type(torch.int8)
            self.rng_idx.data = self.rng_idx.add(input_0.type(torch.long)) % self.depth

            if self.mode == "unipolar":
                return path
            else:
                # for input0 is 0.
                path_inv = (1 - input_0.type(torch.int8)) & (
                            1 - torch.gt(source, self.rng[self.rng_idx_inv]).type(torch.int8))
                # conditional update for rng_idx_inv
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - input_0.type(torch.long)) % self.depth
                return path | path_inv

    def forward(self, input_0, input_1=None):
        return self.FSUMul_forward(input_0, input_1).type(self.stype)


def main():
    sobol_1=  [0,16,24,8,12,28,20,4,6,22,30,14,10,26,18,2,3,19,27,11,15,31,23,7,5,21,29,13,9,25,17,1]
    sobol_2=  [0,16,8,24,12,28,4,20,10,26,2,18,6,22,14,30,15,31,7,23,3,19,11,27,5,21,13,29,9,25,1,17]
    sobol_3=  [0,16,8,24,20,4,28,12,30,14,22,6,10,26,2,18,15,31,7,23,27,11,19,3,17,1,25,9,5,21,13,29]
    sobol_4=  [0,16,8,24,28,12,20,4,14,30,6,22,18,2,26,10,21,5,29,13,9,25,1,17,27,11,19,3,7,23,15,31]
    sobolSeq = torch.tensor(sobol_1)
    '''
        def __init__(self,
                 bitwidth=8,
                 mode="bipolar",
                 static=False,
                 input_prob_1=None,
                 rtype=torch.float,
                 stype=torch.float):
                 '''
    testSample = SC_Based_Mul(bitwidth=8,mode='unipolar',)

if __name__ == "__main__":
    main()