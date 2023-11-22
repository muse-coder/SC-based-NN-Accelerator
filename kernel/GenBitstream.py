import torch
# from stream.gen import RNG, SourceGen, BSGen
# from kernel.shiftreg import ShiftReg
import math
import sys
import time
# class GenBitstream(torch.nn.Module):
#     """
#     Compare source data with rng_seq[rng_idx] to generate bit streams from source
#     only one rng sequence is used here
#     """
#
#     def __init__(self,
#                  rngSeq = [],device = "cuda:0" ):
#         super(GenBitstream, self).__init__()
#         # self.source_data = source_data
#         self.seqLenth = rngSeq.size(0)
#         self.device = device
#         assert rngSeq != None , "random number sequence should not be None"
#         self.RngSeq = rngSeq
#
#     def forward(self, inputData,dataWidth = 8):
#         len = self.RngSeq.size(0)
#         quantizedata = inputData / (2 ** (dataWidth - math.log2(len)))
#         # if(quantizedata-math.floor(quantizedata) >=0.5 ):
#         #     New_quantizedata = math.ceil(quantizedata)
#         # else:
#         #     New_quantizedata = math.floor(quantizedata)
#         #
#         # if (New_quantizedata==0):
#         #     return torch.zeros((len,)).to(self.device)
#
#         New_quantizedata = round(quantizedata)
#
#         sourceDataSeq = torch.round(torch.full((len,) , New_quantizedata )).to(torch.int).to(self.device)
#
#
#
#
#         bitstream = (sourceDataSeq > self.RngSeq).int()
#         return bitstream

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


def tensorGenBitstreamSeries(rngSeq,tensorInputData,index,dataWidth = 8  ):
    len = rngSeq.size(0)
    quantizeData = (torch.round(tensorInputData / (2 ** (dataWidth - math.log2(len))))).to(rngSeq.device)
    # quantizeDataMul = quantizeData.unsqueeze(2)
    # rngSeqMul = rngSeq.unsqueeze(0).unsqueeze(1)
    singleBitstreamMul = (quantizeData> rngSeq[index]).int()


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
    binary_strings = [format(int(num.item()), 'b') for num in tensor.reshape(-1)]

    # 计算每个二进制字符串的有效位数
    significant_bits = [len(binary_string) for binary_string in binary_strings]

    # 将有效位数还原为与输入张量相同的形状
    significant_bits_tensor = torch.tensor(significant_bits).view(tensor.shape)
    result = (significant_bits_tensor - 1).to(tensor.device)
    return result

def TensorLeftShiftBits(data,dataWidth):
    # 将张量转换为整数类型（如果是浮点数）
    dataExceptZero = torch.where(data>0 , data, 2**dataWidth-1)
    dividedData = (2**dataWidth-1)/dataExceptZero
    log2Result =torch.log2(dividedData)
    log2ResultFloor = torch.floor(log2Result)
    return log2ResultFloor




def EnlargeModule(originalData, dataWidth):
    if originalData == 0:
        return 0,0
    # binary_str = format(originalData, f"0{dataWidth}b")
    leftShiftTime = dataWidth -  FindHighestOne(originalData,dataWidth) - 1
    enlargedNumber = int(originalData.item()) << leftShiftTime

    return enlargedNumber , leftShiftTime

def TensorEnlargeModule(tensorData, dataWidth):
    # leftShiftTimeTensor = dataWidth - TensorFindHighestOne(tensorData) - 1
    leftShiftTimeTensor = TensorLeftShiftBits(data= tensorData , dataWidth= dataWidth)
    enlargedNumberTensor = tensorData *(2**leftShiftTimeTensor)

    return enlargedNumberTensor , leftShiftTimeTensor

def BitstreamMUL(bitstream_1,bitstream_2,leftshit_1,leftshit_2,rngSeqLengthLog,dataWidth):
    resultBitstream = (bitstream_1.int() & bitstream_2.int())
    resultSum = resultBitstream.sum()
    resultBinary = (resultSum * (2**(2*dataWidth-rngSeqLengthLog-leftshit_2-leftshit_1)))
    return torch.tensor(resultBinary)

# def GenBitstreamGroup (originData_1, rngSeq , dataWidth , device):
#     Zero = 0
#     if originData_1==0:
#         Zero = 1
#     enlargedData_1, leftShift_1 = EnlargeModule(originalData=originData_1, dataWidth=dataWidth)
#     testSample_1 = GenBitstream(rngSeq=rngSeq).to(device)
#     bitstream_1 = testSample_1(enlargedData_1, dataWidth=dataWidth).to(device)
#     return bitstream_1 , leftShift_1 ,Zero

# def SC_MUL(originData_1 , originData_2 , rngSeq , dataWidth , device):
#     bitstreamLength = len(rngSeq)
#     ascendingSeq = torch.tensor([x for x in range(bitstreamLength)]).to(device)
#     enlargedData_1, leftShift_1 = EnlargeModule(originalData=originData_1,dataWidth= dataWidth)
#     enlargedData_2, leftShift_2 = EnlargeModule(originalData=originData_2, dataWidth=dataWidth)
#     testSample_1 = GenBitstream(rngSeq=rngSeq).to(device)
#     testSample_2 = GenBitstream(rngSeq=ascendingSeq).to(device)
#     bitstream_1 = testSample_1(enlargedData_1 ,dataWidth =  dataWidth).to(device)
#     bitstream_2 = testSample_2(enlargedData_2 ,dataWidth =  dataWidth).to(device)
    # print(bitstream_1.tolist())
    # print(bitstream_2.tolist())

    resultBinary = BitstreamMUL (bitstream_1,bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog = math.log2(bitstreamLength) ,dataWidth=dataWidth).to(device)
    # print(1-resultBinary/(originData_1*originData_2))
    return resultBinary

def getMemorySpace(tensor_variable):

# 获取Tensor数据占用的内存大小（以字节为单位）
    data_size_in_bytes = tensor_variable.element_size() * tensor_variable.numel()

# 将字节大小转换为更常见的单位，如千兆字节（MB）
    data_size_in_megabytes = data_size_in_bytes / (1024**2)
    return data_size_in_megabytes

def matrixMulSC(tensorData_1 , tensorData_2 , rngSeq , dataWidth , device):
    startTime = time.time()
    bitstreamLength = len(rngSeq)
    ascendingSeq = torch.tensor([x for x in range(bitstreamLength)]).to(device)
    enlargedData_1 , dataLeftShiftTime_1 =  TensorEnlargeModule(tensorData=abs(tensorData_1), dataWidth=dataWidth)
    enlargedData_2 , dataLeftShiftTime_2 =  TensorEnlargeModule(tensorData=abs(tensorData_2), dataWidth=dataWidth)
    dataShape_1 = tensorData_1.size()
    dataShape_2 = tensorData_2.size()
    signData_1 =  torch.sign(tensorData_1)
    signData_2 =  torch.sign(tensorData_2)
    '''
    Begin:将数据维度转换成合适shape
    '''
    dataLeftShiftTime_1 = (dataLeftShiftTime_1.unsqueeze(1)).repeat(1,dataShape_2[1],1)
    dataLeftShiftTime_2 = (dataLeftShiftTime_2.unsqueeze(0)).repeat(dataShape_1[0],1,1)
    dataLeftShiftTime_2 = torch.transpose(input=dataLeftShiftTime_2,dim0=1,dim1=2)
    dataScaledTime =  2*dataWidth -( dataLeftShiftTime_1 + dataLeftShiftTime_2) - math.log2(bitstreamLength)
    del dataLeftShiftTime_1
    del dataLeftShiftTime_2

    tensorBit_1 = tensorGenBitstreamMulti(rngSeq = rngSeq , tensorInputData= enlargedData_1 , dataWidth= dataWidth).to(device)
    tensorBit_2 = tensorGenBitstreamMulti(rngSeq = ascendingSeq , tensorInputData= enlargedData_2 , dataWidth= dataWidth).to(device)
    tensorBit_1 = tensorBit_1.to(torch.float16)
    tensorBit_2 = tensorBit_2.to(torch.float16)
    torch.mul(input=tensorBit_1, other=(signData_1.unsqueeze(2).repeat(1,1,bitstreamLength)),out=tensorBit_1)
    torch.mul(input=tensorBit_2, other=(signData_2.unsqueeze(2).repeat(1, 1, bitstreamLength)), out=tensorBit_2)

    del signData_1
    del signData_2
    # tensorBit_1 = (tensorBit_1.unsqueeze(1)).repeat(1,dataShape_2[1],1,1)
    # tensorBit_2 = (tensorBit_2.unsqueeze(0)).repeat(dataShape_1[0], 1, 1,1)
    # tensorBit_2 = torch.transpose(input=tensorBit_2,dim0=1,dim1=2)
    # tensorBit_2 = torch.transpose(input=tensorBit_2 ,dim0=2,dim1=3)
    '''
        End:将数据维度转换成合适shape
    '''
    # tensorBit_1_old = (tensorBit_1.unsqueeze(1)).repeat(1,dataShape_2[1],1,1)
    # tensorBit_2_old = (tensorBit_2.unsqueeze(0)).repeat(dataShape_1[0], 1, 1,1)


    tensorBit_1 = tensorBit_1.unsqueeze(1).expand(-1, dataShape_2[1],-1,-1)
    tensorBit_2 = tensorBit_2.unsqueeze(0).expand(dataShape_1[0], -1, -1, -1)
    tensorBit_2 = tensorBit_2.transpose(1, 2).transpose(2, 3)

    # 执行矩阵乘法

    SCResult = (tensorBit_1).matmul(tensorBit_2)

    del tensorBit_1
    del tensorBit_2

    SCResultDiagonal =  torch.diagonal(input= SCResult,dim1=2,dim2=3)
    SCResultDiagonal = SCResultDiagonal.mul(2**dataScaledTime)
    SCMatrixResult = torch.sum(input=SCResultDiagonal,dim=2)
    # print(SCMatrixResult)
    endTime = time.time()
    print(f"Parallel MulSC cost time : {endTime - startTime}")

    return SCMatrixResult

    # exactResult =


def splitTensor(tensorA, num_slices):
    m, n = tensorA.shape

    # 计算每个子张量的宽度
    width = n // num_slices

    # 初始化一个空列表，用于存储切片后的子张量
    sliced_tensors = []

    # 循环生成切片
    for i in range(num_slices):
        start_col = i * width
        end_col = (i + 1) * width if i < num_slices - 1 else n
        sliced_tensor = tensorA[:, start_col:end_col]
        sliced_tensors.append(sliced_tensor)

    return sliced_tensors



def matrixMacSeperate(tensorData_1 , tensorData_2 ,num_slices, rngSeq , dataWidth , device):
    dataShape_1 = tensorData_1.size()
    dataShape_2 = tensorData_2.size()
    subTensorSet = splitTensor(tensorA= tensorData_2, num_slices= num_slices)
    accumulateRes = torch.zeros(dataShape_1[0],dataShape_2[1]).to(device)

    # 计算每个子张量的宽度
    width = dataShape_2[1] // num_slices
    # 循环生成切片
    for (i  , subTensor )in  enumerate(subTensorSet):
        start_col = i * width
        end_col = (i + 1) * width if i < num_slices - 1 else dataShape_2[1]
        accumulateRes[:, start_col:end_col] = matrixMulSeriesSC(tensorData_1=tensorData_1 , tensorData_2= subTensor,rngSeq= rngSeq , dataWidth= dataWidth, device=device)

    return  accumulateRes


def matrixMulSeriesSC(tensorData_1 , tensorData_2 , rngSeq , dataWidth , device):
    startTime =  time.time()
    bitstreamLength = len(rngSeq)
    ascendingSeq = torch.tensor([x for x in range(bitstreamLength)]).to(device)
    enlargedData_1 , dataLeftShiftTime_1 =  TensorEnlargeModule(tensorData=abs(tensorData_1), dataWidth=dataWidth)
    enlargedData_2 , dataLeftShiftTime_2 =  TensorEnlargeModule(tensorData=abs(tensorData_2), dataWidth=dataWidth)
    dataShape_1 = tensorData_1.size()
    dataShape_2 = tensorData_2.size()
    signData_1 =  torch.sign(tensorData_1)
    signData_2 =  torch.sign(tensorData_2)
    '''
    Begin:将数据维度转换成合适shape
    '''
    dataLeftShiftTime_1 = (dataLeftShiftTime_1.unsqueeze(1)).repeat(1,dataShape_2[1],1)
    dataLeftShiftTime_2 = (dataLeftShiftTime_2.unsqueeze(0)).repeat(dataShape_1[0],1,1)
    dataLeftShiftTime_2 = torch.transpose(input=dataLeftShiftTime_2,dim0=1,dim1=2)
    dataScaledTime =  2*dataWidth -( dataLeftShiftTime_1 + dataLeftShiftTime_2 ) - math.log2(bitstreamLength)

    # SCResult = torch.empty((dataShape_1[0],dataShape_2[1]),dtype=torch.float)
    SCBitACC = torch.zeros((dataShape_1[0],dataShape_2[1],dataShape_2[0]),dtype=torch.float).to(device)
    for i in range (bitstreamLength):
        # print(i)
        tensorBit_1 = tensorGenBitstreamSeries(rngSeq = rngSeq , tensorInputData= enlargedData_1 , index= i , dataWidth= dataWidth).to(device)
        tensorBit_2 = tensorGenBitstreamSeries(rngSeq = ascendingSeq , tensorInputData= enlargedData_2 ,index= i , dataWidth= dataWidth).to(device)
        tensorBit_1 = tensorBit_1.to(torch.float16)
        tensorBit_2 = tensorBit_2.to(torch.float16)
        torch.mul(input=tensorBit_1, other=(signData_1),out=tensorBit_1)
        torch.mul(input=tensorBit_2, other=(signData_2), out=tensorBit_2)
        tensorBit_1 = (tensorBit_1.unsqueeze(1)).repeat(1,dataShape_2[1],1)
        tensorBit_2 = (tensorBit_2.unsqueeze(0)).repeat(dataShape_1[0],1,1)
        tensorBit_2 = torch.transpose(input=tensorBit_2,dim0=1,dim1=2)
        SCBitACC    = SCBitACC + tensorBit_1 * tensorBit_2
        # tensorBit_2 = torch.transpose(input=tensorBit_2 ,dim0=1,dim1=2)
    SCBitACC =  SCBitACC.mul(2** dataScaledTime)
    del tensorBit_1
    del tensorBit_2
    del dataScaledTime
    SCResult = torch.sum(input=SCBitACC,dim=2)
    # print(SCResult )
    endTime = time.time()
    print(f"SeriesSC cost time : {endTime - startTime}")
    return SCResult
    # return SCMatrixResult


def matrixMulSeriesSC_new(tensorData_1 , tensorData_2 , rngSeq , dataWidth , device):
    startTime =  time.time()
    bitstreamLength = len(rngSeq)
    ascendingSeq = torch.tensor([x for x in range(bitstreamLength)]).to(device)
    enlargedData_1 , dataLeftShiftTime_1 =  TensorEnlargeModule(tensorData=abs(tensorData_1), dataWidth=dataWidth)
    enlargedData_2 , dataLeftShiftTime_2 =  TensorEnlargeModule(tensorData=abs(tensorData_2), dataWidth=dataWidth)
    dataShape_1 = tensorData_1.size()
    dataShape_2 = tensorData_2.size()
    signData_1 =  torch.sign(tensorData_1)
    signData_2 =  torch.sign(tensorData_2)
    '''
    Begin:将数据维度转换成合适shape
    '''
    # dataScaledTime =  2*dataWidth -( dataLeftShiftTime_1 + dataLeftShiftTime_2 ) - math.log2(bitstreamLength)
    dataScaledFactor_1 = 2**(dataWidth - dataLeftShiftTime_1 - math.log2(bitstreamLength))
    dataScaledFactor_2 = 2**(dataWidth - dataLeftShiftTime_2)
    # SCResult = torch.empty((dataShape_1[0],dataShape_2[1]),dtype=torch.float)
    SCBitACC = torch.zeros((dataShape_1[0],dataShape_2[1]),dtype=torch.float).to(device)
    assert torch.isnan(SCBitACC).any() == False

    for i in range (bitstreamLength):
        # print(i)
        tensorBit_1 = tensorGenBitstreamSeries(rngSeq = rngSeq , tensorInputData= enlargedData_1 , index= i , dataWidth= dataWidth).to(device)
        tensorBit_2 = tensorGenBitstreamSeries(rngSeq = ascendingSeq , tensorInputData= enlargedData_2 ,index= i , dataWidth= dataWidth).to(device)
        tensorBit_1 = tensorBit_1.to(torch.float16)
        tensorBit_2 = tensorBit_2.to(torch.float16)
        torch.mul(input=tensorBit_1, other=(signData_1),out=tensorBit_1)
        torch.mul(input=tensorBit_2, other=(signData_2), out=tensorBit_2)
        tensorBit_1 = tensorBit_1.mul(dataScaledFactor_1)
        tensorBit_2 = tensorBit_2.mul(dataScaledFactor_2)
        SCBitACC    = SCBitACC + tensorBit_1.matmul(tensorBit_2)
        assert torch.isinf(tensorBit_1).any() == False
        assert torch.isinf(tensorBit_2).any() == False
        test = torch.isinf(tensorBit_1.matmul(tensorBit_2)).any()
        if test:
            print(test)
        if (tensorBit_1.matmul(tensorBit_2)).abs().max().log2()>16:
            print("inf")
        assert (torch.isinf(tensorBit_1.matmul(tensorBit_2)).any()) == False
        assert torch.isinf(SCBitACC).any() == False
        assert torch.isnan(tensorBit_1).any() == False
        assert torch.isnan(tensorBit_2).any() == False
        assert torch.isnan(tensorBit_1.matmul(tensorBit_2)).any() == False

        assert torch.isnan(SCBitACC    ).any() == False

        # tensorBit_2 = torch.transpose(input=tensorBit_2 ,dim0=1,dim1=2)

    # SCResult =   SCBitACC * (2 ** (-math.log2(bitstreamLength)))
    SCResult = SCBitACC
    del tensorBit_1
    del tensorBit_2

    endTime = time.time()
    # print(f"SeriesSCNew cost time : {endTime - startTime}")
    SCResult = SCResult.to(torch.float32)
    assert torch.isnan(SCResult).any() ==False
    assert torch.isinf(SCResult).any() ==False
    return SCResult



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

    # resultBinary = BitstreamMUL (bitstream_1,bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog = math.log2(bitstreamLength) ,dataWidth=dataWidth).to(device)
    # print(1-resultBinary/(originData_1*originData_2))

    for i in range(enlargedData_1.size(0)):
        for j in range(enlargedData_2.size(1)):
            tensorResult[i, j, :] += (dataB + dataA) * (a & b)

    tensorResult
    return enlargedData_2



#
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sobol_1 = [0, 16, 24, 8, 12, 28, 20, 4, 6, 22, 30, 14, 10, 26, 18, 2, 3, 19, 27, 11, 15, 31, 23, 7, 5, 21, 29, 13,
               9, 25, 17, 1]

    sobol_1 = [0,8,12,4,6,14,10,2,3,11,15,7,5,13,9,1]
    sobolTensor = torch.tensor(sobol_1).to(device)
    dataWidth = 16
    for i in range (10):
        tensor1 = torch.randint(-255,255, size=(int(10240), 256)).to(device)
        tensor2 = torch.randint(-255,255, size=(256, 64)).to(device)
        print("***********")
        # approximateResult_2 = matrixMulSeriesSC(tensorData_1=tensor1 , tensorData_2= tensor2, rngSeq=sobolTensor ,dataWidth=8 ,device= device)
        # approximateResult_1 = matrixMacSeperate(tensorData_1=tensor1 , tensorData_2= tensor2, num_slices= 4 ,  rngSeq=sobolTensor ,dataWidth=8 ,device= device)
        approximateResult_3 = matrixMulSeriesSC_new(tensorData_1=tensor1, tensorData_2=tensor2, rngSeq=sobolTensor,
                                                dataWidth=dataWidth, device=device)
        approximateResult_2 = matrixMulSeriesSC_new(tensorData_1=tensor1, tensorData_2=tensor2, rngSeq=sobolTensor,
                                                dataWidth=8, device=device)

        print("***********\n\n")
        # assert torch.equal(approximateResult_1,approximateResult_2)
        assert torch.equal(approximateResult_2,approximateResult_3)
        exactResutl = tensor1.to(torch.float).matmul((tensor2).to(torch.float))
        relativeError = abs(1 - (approximateResult_3 / exactResutl))
        absoluteError = abs(exactResutl - approximateResult_3 )
        maxRED,index1 = torch.max(input=relativeError) , torch.argmax(input=relativeError)
        minRED,index2 = torch.min(input=relativeError) , torch.argmin(input=relativeError)
        maxAED,index1 = torch.max(input=absoluteError) , torch.argmax(input=absoluteError)
        minAED,index2 = torch.min(input=absoluteError) , torch.argmin(input=absoluteError)
        non_zero_RED_index = torch.argwhere(input= relativeError)
        non_zero_RED =relativeError[non_zero_RED_index]
        maxRED, index1 = torch.max(input=non_zero_RED), torch.argmax(input=non_zero_RED)
        minRED, index2 = torch.min(input=non_zero_RED), torch.argmin(input=non_zero_RED)
    #


    print(maxRED)
    print(minRED)