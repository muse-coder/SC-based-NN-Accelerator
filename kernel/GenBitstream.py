import torch
from stream.gen import RNG, SourceGen, BSGen
from kernel.shiftreg import ShiftReg
import math
class GenBitstream(torch.nn.Module):
    """
    Compare source data with rng_seq[rng_idx] to generate bit streams from source
    only one rng sequence is used here
    """

    def __init__(self,
                 rngSeq = [],device = "cuda:0" ):
        super(GenBitstream, self).__init__()
        # self.source_data = source_data
        self.seqLenth = rngSeq.size(0)
        self.device = device
        assert rngSeq != None , "random number sequence should not be None"
        self.RngSeq = rngSeq

    def forward(self, inputData,dataWidth = 8):
        len = self.RngSeq.size(0)
        quantizedata = inputData / (2 ** (dataWidth - math.log2(len)))
        if (quantizedata==0):
            return torch.zeros((len,)).to(self.device)
        sourceDataSeq = torch.full((len,) , quantizedata ).to(self.device)
        bitstream = (sourceDataSeq > self.RngSeq).int()
        return bitstream

def FindHighestOne(num, dataWidth):
    mask = 1 << (dataWidth - 1)  # 创建一个掩码，将其移到最高位
    for position in range(dataWidth - 1, -1, -1):
        if num & mask:
            return position
        mask >>= 1  # 右移掩码，检查下一位
    return None

def EnlargeModule(originalData, dataWidth):
    if originalData == 0:
        return 0,0
    binary_str = format(originalData, f"0{dataWidth}b")
    leftShiftTime = dataWidth -  FindHighestOne(originalData,dataWidth) - 1
    enlargedNumber = originalData << leftShiftTime

    return enlargedNumber , leftShiftTime

def BitstreamMUL(bitstream_1,bitstream_2,leftshit_1,leftshit_2,rngSeqLengthLog,dataWidth):
    resultBitstream = (bitstream_1.int() & bitstream_2.int())
    resultSum = resultBitstream.sum()
    resultBinary = resultSum * (2**(2*dataWidth-rngSeqLengthLog-leftshit_2-leftshit_1))
    return resultBinary

def GenBitstreamGroup (originData_1, rngSeq , dataWidth , device):
    Zero = 0
    if originData_1==0:
        Zero = 1
    enlargedData_1, leftShift_1 = EnlargeModule(originalData=originData_1, dataWidth=dataWidth)
    testSample_1 = GenBitstream(rngSeq=rngSeq).to(device)
    bitstream_1 = testSample_1(enlargedData_1, dataWidth=dataWidth).to(device)
    return bitstream_1 , leftShift_1 ,Zero

def SC_MUL(originData_1 , originData_2 , rngSeq , dataWidth , device):
    bitstreamLength = len(rngSeq)
    ascendingSeq = torch.tensor([x for x in range(bitstreamLength)]).to(device)
    enlargedData_1, leftShift_1 = EnlargeModule(originalData=originData_1,dataWidth= dataWidth)
    enlargedData_2, leftShift_2 = EnlargeModule(originalData=originData_2, dataWidth=dataWidth)
    testSample_1 = GenBitstream(rngSeq=rngSeq).to(device)
    testSample_2 = GenBitstream(rngSeq=ascendingSeq).to(device)
    bitstream_1 = testSample_1(enlargedData_1 ,dataWidth =  dataWidth).to(device)
    bitstream_2 = testSample_2(enlargedData_2 ,dataWidth =  dataWidth).to(device)
    resultBinary = BitstreamMUL (bitstream_1,bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog = math.log2(bitstreamLength) ,dataWidth=dataWidth).to(device)
    # print(1-resultBinary/(originData_1*originData_2))
    return resultBinary
if __name__ == "__main__":
    sobol_1 = [0, 16, 24, 8, 12, 28, 20, 4, 6, 22, 30, 14, 10, 26, 18, 2, 3, 19, 27, 11, 15, 31, 23, 7, 5, 21, 29, 13,
               9, 25, 17, 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sobolTensor = torch.tensor(sobol_1).to(device)
    result = SC_MUL(originData_1=33, originData_2=44, rngSeq=sobolTensor, dataWidth=8, device=device)
    print(result)