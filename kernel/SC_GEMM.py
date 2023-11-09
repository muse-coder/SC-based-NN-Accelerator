import torch
from kernel.GenBitstream import  *
import math

class BitstreamSource:
    def __init__(self,
                 dataWidth,
                 rngSeq,
                 device="cuda:0"):
        self.dataWidth = dataWidth
        self.rngSeq = rngSeq.to(device)
        self.bitstreamlength = len(rngSeq)
        self.ascendingSeq  = torch.tensor([x for x in range(self.bitstreamlength)]).to(device)
        self.device = device
        assert rngSeq != None , "random number sequence should not be None"

        self.BitstreamSource = torch.zeros( 2 ** dataWidth, len(self.rngSeq)).to(self.device)
        self.LeftShiftZeroSource = torch.zeros(2 ** dataWidth,2).to(self.device)

        for i in range (2**dataWidth):
            data = torch.tensor(i)
            bitstream, leftShift, Zero = GenBitstreamGroup(originData_1= data ,
                                                             rngSeq=rngSeq, dataWidth=self.dataWidth,
                                                             device=self.device)
            self.BitstreamSource[i,:] = bitstream
            self.LeftShiftZeroSource[i,0] = leftShift
            self.LeftShiftZeroSource[i,1] = Zero

    def GetBitstream(self,i):
        return self.BitstreamSource[i,:] , self.LeftShiftZeroSource[i,0] , self.LeftShiftZeroSource[i,1]


class NewSCbasedGEMM:
    def __init__(self,
                 tensor_1,
                 tensor_2,
                 dataWidth,
                 rngSeq,
                 device = "cuda:0" ):

        self.tensor_1 = tensor_1.to(device)
        self.tensor_2 = tensor_2.to(device)
        self.rngSeq = rngSeq.to(device)
        self.bitstreamlength = len(rngSeq)
        self.ascendingSeq  = torch.tensor([x for x in range(self.bitstreamlength)]).to(device)
        self.device = device
        self.dataWidth = dataWidth
        assert rngSeq != None , "random number sequence should not be None"

        self.rows1, self.cols1 = self.tensor_1.size()
        self.rows2, self.cols2 = self.tensor_2.size()
        # 执行矩阵乘法

    def calculate(self,BitstreamSourceInstanceA,BitstreamSourceInstanceB):
        approximateResult = torch.zeros(self.rows1, self.cols2).to(self.device)
        # exactResult = torch.zeros(self.rows1, self.cols2).to(self.device)
        print(f"rows1:{self.rows1}, cols2:{self.cols2}, cols1:{self.cols1},  ")
        for i in range(self.rows1):
            print(f"rows now :{i}")
            for j in range(self.cols2):
                for k in range(self.cols1):
                    originalData_1 = (self.tensor_1 [i,k]).to(torch.int)
                    if (originalData_1<0):
                        sign_1 = -1
                    else :
                        sign_1 = 1

                    bitstream_1 , leftShift_1 , zero_1= BitstreamSourceInstanceA.GetBitstream(abs(originalData_1))

                    originalData_2 = (self.tensor_2 [k,j]).to(torch.int)
                    if (originalData_2<0):
                        sign_2 = -1
                    else :
                        sign_2 = 1

                    bitstream_2 , leftShift_2 , zero_2= BitstreamSourceInstanceB.GetBitstream(abs(originalData_2))

                    if (zero_1==1 or zero_2==1):
                        signSCResult = torch.tensor(0)
                    else:
                        absSCResult = BitstreamMUL(bitstream_1, bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog=math.log2(self.bitstreamlength), dataWidth=self.dataWidth).to(self.device)
                        signSCResult = absSCResult * sign_2 * sign_1


                    approximateResult[i, j] += signSCResult

        return approximateResult




    def ParallelCalculate(self,BitstreamSourceInstanceA,BitstreamSourceInstanceB):
        approximateResult = torch.zeros(self.rows1, self.cols2).to(self.device)

        newTensor_1 = self.tensor_1.unsequeeze(1)

        print(f"rows1:{self.rows1}, cols2:{self.cols2}, cols1:{self.cols1},  ")
        for i in range(self.rows1):
                for j in range(self.cols1):
                    originalData_1 = (self.tensor_1 [i,j]).to(torch.int)
                    if (originalData_1<0):
                        sign_1 = -1
                    else :
                        sign_1 = 1

                    bitstream_1 , leftShift_1 , zero_1= BitstreamSourceInstanceA.GetBitstream(abs(originalData_1))

                    originalData_2 = (self.tensor_2 [k,j]).to(torch.int)
                    if (originalData_2<0):
                        sign_2 = -1
                    else :
                        sign_2 = 1

                    bitstream_2 , leftShift_2 , zero_2= BitstreamSourceInstanceB.GetBitstream(abs(originalData_2))

                    if (zero_1==1 or zero_2==1):
                        signSCResult = torch.tensor(0)
                    else:
                        absSCResult = BitstreamMUL(bitstream_1, bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog=math.log2(self.bitstreamlength), dataWidth=self.dataWidth).to(self.device)
                        signSCResult = absSCResult * sign_2 * sign_1


                    approximateResult[i, j] += signSCResult

        return approximateResult




if __name__ == "__main__":
    sobol_1 = [0, 16, 24, 8, 12, 28, 20, 4, 6, 22, 30, 14, 10, 26, 18, 2, 3, 19, 27, 11, 15, 31, 23, 7, 5, 21, 29, 13,
               9, 25, 17, 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sobolTensor = torch.tensor(sobol_1).to(device)
    tensor1 = torch.randint(-128, 128, size=(128, 128))
    tensor2 = torch.randint(-128, 128, size=(128, 4))
    ascendingSeq = torch.tensor([x for x in range(len(sobol_1))]).to(device)

    # tensor1 = torch.randint(-32, 32, size=(128, 32))
    # tensor2 = torch.randint(-32, 32, size=(32, 4))
    SobolBitstreamSource = BitstreamSource(dataWidth=8,rngSeq=sobolTensor,device=device)
    AscendingBitstreamSource = BitstreamSource(dataWidth=8, rngSeq=ascendingSeq, device=device)

    NewSCbasedGEMMInstance = NewSCbasedGEMM(tensor_1=tensor1,tensor_2=tensor2,dataWidth=8,rngSeq=sobolTensor,device=device)
    approximateResult = NewSCbasedGEMMInstance.calculate(BitstreamSourceInstanceA=SobolBitstreamSource, BitstreamSourceInstanceB= AscendingBitstreamSource)

    result = torch.matmul(tensor1, tensor2).to(device)
    relativeError = abs(1 - approximateResult / result)
    print(relativeError)