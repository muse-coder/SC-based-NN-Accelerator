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
        # if(quantizedata-math.floor(quantizedata) >=0.5 ):
        #     New_quantizedata = math.ceil(quantizedata)
        # else:
        #     New_quantizedata = math.floor(quantizedata)
        #
        # if (New_quantizedata==0):
        #     return torch.zeros((len,)).to(self.device)

        New_quantizedata = round(quantizedata)

        sourceDataSeq = torch.round(torch.full((len,) , New_quantizedata )).to(torch.int).to(self.device)




        bitstream = (sourceDataSeq > self.RngSeq).int()
        return bitstream

def TensorGenBitstream(rngSeq,tensorInputData,index,dataWidth = 8 ):
    len = rngSeq.size(0)
    quantizedata = (torch.round(tensorInputData / (2 ** (dataWidth - math.log2(len))))).to(rngSeq.device)


    singleBitstream = (quantizedata> rngSeq[index]).int()
    return singleBitstream

def tensorGenBitstreamMulti(rngSeq,tensorInputData,dataWidth = 8  ):
    len = rngSeq.size(0)
    quantizeData = (torch.round(tensorInputData / (2 ** (dataWidth - math.log2(len))))).to(rngSeq.device)
    quantizeDataMul = quantizeData.unsqueeze(2)
    rngSeqMul = rngSeq.unsqueeze(0).unsqueeze(1)
    singleBitstreamMul = (quantizeDataMul> rngSeqMul).int()
    quantizeData_T = torch.transpose(input=quantizeData,dim0=0,dim1=1)
    singleBitstreamMul_T = torch.transpose(input=singleBitstreamMul,dim0=0,dim1=1)
    # originalQuantizeData = torch.sum(input = singleBitstreamMul,dim= 2 )


    return singleBitstreamMul




def FindHighestOne(num, dataWidth):
    mask = 1 << (dataWidth - 1)  # 创建一个掩码，将其移到最高位
    data = int(num.item())
    for position in range(dataWidth - 1, -1, -1):
        if data & mask:
            return position
        mask >>= 1  # 右移掩码，检查下一位
    return None


def TensorFindHighestOne(tensor):
    # 将张量转换为整数类型（如果是浮点数）
    tensor = tensor.to(torch.int)

    # 获取张量中每个元素的二进制表示
    binary_strings = [format(int(num.item()), 'b') for num in tensor.view(-1)]

    # 计算每个二进制字符串的有效位数
    significant_bits = [len(binary_string) for binary_string in binary_strings]

    # 将有效位数还原为与输入张量相同的形状
    significant_bits_tensor = torch.tensor(significant_bits).view(tensor.shape)
    result = (significant_bits_tensor - 1).to(tensor.device)
    return result


def EnlargeModule(originalData, dataWidth):
    if originalData == 0:
        return 0,0
    # binary_str = format(originalData, f"0{dataWidth}b")
    leftShiftTime = dataWidth -  FindHighestOne(originalData,dataWidth) - 1
    enlargedNumber = int(originalData.item()) << leftShiftTime

    return enlargedNumber , leftShiftTime

def TensorEnlargeModule(tensorData, dataWidth):
    leftShiftTimeTensor = dataWidth - TensorFindHighestOne(tensorData) - 1
    enlargedNumberTensor = tensorData *(2**leftShiftTimeTensor)

    return enlargedNumberTensor , leftShiftTimeTensor

def BitstreamMUL(bitstream_1,bitstream_2,leftshit_1,leftshit_2,rngSeqLengthLog,dataWidth):
    resultBitstream = (bitstream_1.int() & bitstream_2.int())
    resultSum = resultBitstream.sum()
    resultBinary = (resultSum * (2**(2*dataWidth-rngSeqLengthLog-leftshit_2-leftshit_1)))
    return torch.tensor(resultBinary)

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
    # print(bitstream_1.tolist())
    # print(bitstream_2.tolist())

    resultBinary = BitstreamMUL (bitstream_1,bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog = math.log2(bitstreamLength) ,dataWidth=dataWidth).to(device)
    # print(1-resultBinary/(originData_1*originData_2))
    return resultBinary

def matrixMulSC(tensorData_1 , tensorData_2 , rngSeq , dataWidth , device):
    bitstreamLength = len(rngSeq)
    ascendingSeq = torch.tensor([x for x in range(bitstreamLength)]).to(device)
    enlargedData_1 , dataLeftShiftTime_1 =  TensorEnlargeModule(tensorData=tensorData_1, dataWidth=dataWidth)
    enlargedData_2 , dataLeftShiftTime_2 =  TensorEnlargeModule(tensorData=tensorData_2, dataWidth=dataWidth)
    dataShape_1 = tensorData_1.size()
    dataShape_2 = tensorData_2.size()
    signData_1 =  torch.sign(tensorData_1)
    signData_2 =  torch.sign(tensorData_2)
    dataLeftShiftTime_1_expand = (dataLeftShiftTime_1.unsqueeze(1)).repeat(1,dataShape_2[1],1)
    dataLeftShiftTime_2_expand = (dataLeftShiftTime_2.unsqueeze(0)).repeat(dataShape_1[0],1,1)
    dataLeftShiftTime_2_expand_reshape = torch.transpose(input=dataLeftShiftTime_2_expand,dim0=1,dim1=2)
    dataScaledTime =  2*dataWidth -( dataLeftShiftTime_1_expand + dataLeftShiftTime_2_expand_reshape) - math.log2(bitstreamLength)

    tensorBit_1 = tensorGenBitstreamMulti(rngSeq = rngSeq , tensorInputData= enlargedData_1 , dataWidth= dataWidth).to(device)
    tensorBit_2 = tensorGenBitstreamMulti(rngSeq = ascendingSeq , tensorInputData= enlargedData_2 , dataWidth= dataWidth).to(device)
    tensorBitSign_1 = tensorBit_1 * (signData_1.unsqueeze(2).repeat(1,1,bitstreamLength))
    tensorBitSign_2 = tensorBit_2 * (signData_2.unsqueeze(2).repeat(1,1,bitstreamLength))


    tensorBit_1_expand = (tensorBitSign_1.unsqueeze(1)).repeat(1,dataShape_2[1],1,1)
    tensorBit_2_expand = (tensorBitSign_2.unsqueeze(0)).repeat(dataShape_1[0], 1, 1,1)
    tensorBit_2_expand_reshape = torch.transpose(input=tensorBit_2_expand,dim0=1,dim1=2)
    tensorBit_2_expand_reshape_2 = torch.transpose(input=tensorBit_2_expand_reshape ,dim0=2,dim1=3)




    SCResult = (tensorBit_1_expand.to(torch.float)).matmul( tensorBit_2_expand_reshape_2.to(torch.float) )

    SCResultDiagonal =  torch.diagonal(input= SCResult,dim1=2,dim2=3)
    SCResultDiagonalScaled = SCResultDiagonal.mul(2**dataScaledTime)
    SCResultDiagonalScaledMac = torch.sum(input=SCResultDiagonalScaled,dim=2)
    exactMatrixResult = tensorData_1.to(torch.float).matmul(tensorData_2.to(torch.float))
    # matResult = torch.sum(input=(torch.sum(input=SCResult,dim=1)),dim=2)
    print(SCResultDiagonalScaledMac)
    return SCResultDiagonalScaledMac
    # exactResult =
def TensorSC_MUL(tensorData_1 , tensorData_2 , rngSeq , dataWidth , device):
    bitstreamLength = len(rngSeq)
    ascendingSeq = torch.tensor([x for x in range(bitstreamLength)]).to(device)
    enlargedData_1, leftShift_1 = TensorEnlargeModule(tensorData=abs(tensorData_1),dataWidth= dataWidth)
    enlargedData_2, leftShift_2 = TensorEnlargeModule(tensorData=abs(tensorData_2), dataWidth=dataWidth)
    signTensorData_1 =  torch.sign(tensorData_1)
    signTensorData_2 =  torch.sign(tensorData_1)


    opA = torch.ones_like(leftShift_1) * 2<<(dataWidth - leftShift_1.to(torch.int ) - 1)
    opB = torch.ones_like(leftShift_2) * 2<<(dataWidth - leftShift_2.to(torch.int ) - 1)
    # opResult = opA + opB

    tensorResult = torch.zeros(enlargedData_1.size(0),enlargedData_2.size(1),enlargedData_2.size(0)).to(tensorData_1.device)
    for i in range(bitstreamLength ):
        tensorBitstream_1 = TensorGenBitstream(rngSeq, tensorInputData= enlargedData_1,index= i, dataWidth=dataWidth)
        tensorBitstream_2 = TensorGenBitstream(ascendingSeq, tensorInputData=enlargedData_2, index=i, dataWidth=dataWidth)
        for i in range(tensorBitstream_1.size(0)):
            for j in range(tensorBitstream_2.size(1)):
                a = tensorBitstream_1[i,:]
                b = tensorBitstream_2.t()[j,:]
                dataA = opA[i,:]
                dataB = opB.t()[j,:]
                signA = signTensorData_1[i,:]
                signB = signTensorData_2.t()[j,:]
                tensorResult[i,j,:]  +=  (dataB*signB+dataA*signA) * (a & b)

        # 使用广播进行逐元素相加操作，生成 tensorC
        # tensorC = tensorA.unsqueeze(2) + tensorB.t().unsqueeze(0)
        # are_equal = torch.equal(tensor2_sub1, tensor2_sub2)
        # 执行逐元素与运算
        # tensorC = tensorA_expanded & tensorB_expanded

        andOpResult = tensorResult[:,:,]
    # resultBinary = BitstreamMUL (bitstream_1,bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog = math.log2(bitstreamLength) ,dataWidth=dataWidth).to(device)
    # print(1-resultBinary/(originData_1*originData_2))

    for i in range(enlargedData_1.size(0)):
        for j in range(enlargedData_2.size(1)):
            tensorResult[i, j, :] += (dataB + dataA) * (a & b)

    tensorResult
    return enlargedData_2
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sobol_1 = [0, 16, 24, 8, 12, 28, 20, 4, 6, 22, 30, 14, 10, 26, 18, 2, 3, 19, 27, 11, 15, 31, 23, 7, 5, 21, 29, 13,
               9, 25, 17, 1]
    sobolTensor = torch.tensor(sobol_1).to(device)




    tensor1 = torch.randint(-128, 255, size=(500, 64)).to(device)
    tensor2 = torch.randint(-128, 255, size=(64, 100)).to(device)


    # tensor1Bit =  tensorGenBitstreamMulti(rngSeq=sobolTensor, tensorInputData=tensor1, dataWidth=8)
    result = matrixMulSC(tensorData_1=tensor1 , tensorData_2= tensor2, rngSeq=sobolTensor ,dataWidth=8 ,device= device)

    result =  TensorSC_MUL(tensorData_1= tensor1, tensorData_2= tensor2,rngSeq= sobolTensor, dataWidth = 8, device = device)
    # 调用GetSignificantBits函数来获取有效位数
    # result = TensorEnlargeModule(tensor,8)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # sobolTensor = torch.tensor(sobol_1).to(device)
    # t = round(18.52)
    # result = SC_MUL(originData_1=torch.tensor(37), originData_2=torch.tensor(47), rngSeq=sobolTensor, dataWidth=8, device=device)
    #
