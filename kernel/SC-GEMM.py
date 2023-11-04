import torch
from GenBitstream import *
import math
class SCbasedGEMM(torch.nn.Module):

    def __init__(self,
                 tensor_1,
                 tensor_2,
                 dataWidth,
                 rngSeq,
                 device = "cuda:0" ):
        super(SCbasedGEMM, self).__init__()
        # self.source_data = source_data
        self.tensor_1 = tensor_1.to(device)
        self.tensor_2 = tensor_2.to(device)
        self.rngSeq = rngSeq.to(device)
        self.bitstreamlength = len(rngSeq)
        self.ascendingSeq  = torch.tensor([x for x in range(self.bitstreamlength)]).to(device)
        self.device = device
        self.dataWidth = dataWidth
        assert rngSeq != None , "random number sequence should not be None"

    def forward(self):
        rows1, cols1 = self.tensor_1.size()
        rows2, cols2 = self.tensor_2.size()
        result = torch.zeros(rows1, cols2).to(self.device)
        # 执行矩阵乘法

        BitstreamSource_1 = torch.zeros(rows1, cols1 , len(self.rngSeq)).to(self.device)
        BitstreamSource_2 = torch.zeros(rows2, cols2 , len(self.rngSeq)).to(self.device)
        LeftShiftZeorSource_1 = torch.zeros(rows1, cols1,2 ).to(self.device)
        LeftShiftZeorSource_2 = torch.zeros(rows2, cols2,2 ).to(self.device)
        for i in range(rows1):
            for j in range(cols1):
                bitstream_1,leftShift_1,Zero_1 = GenBitstreamGroup(originData_1=self.tensor_1[i, j],rngSeq=self.rngSeq,dataWidth=self.dataWidth,device=self.device)
                BitstreamSource_1[i,j,:] = bitstream_1
                LeftShiftZeorSource_1[i,j,0] = leftShift_1
                LeftShiftZeorSource_1[i,j,1] = Zero_1

        for i in range(rows2):
            for j in range(cols2):
                bitstream_2,leftShift_2,Zero_2 = GenBitstreamGroup(originData_1=self.tensor_2[i, j],rngSeq=self.ascendingSeq,dataWidth=self.dataWidth,device=self.device)
                BitstreamSource_2[i, j, :] = bitstream_2
                LeftShiftZeorSource_2[i,j,0] = leftShift_2
                LeftShiftZeorSource_2[i,j,1] = Zero_2

        for i in range(rows1):
            for j in range(cols2):
                for k in range(cols1):
                    bitstream_1 = BitstreamSource_1[i,k,:]
                    leftShift_1 = LeftShiftZeorSource_1[i,k,0]
                    Zero_1 = LeftShiftZeorSource_1[i,k,1].item()
                    originalData_1 = self.tensor_1 [i,k]
                    bitstream_2 = BitstreamSource_2[k,j,:]
                    originalData_2 = self.tensor_2 [k,j]
                    leftShift_2 = LeftShiftZeorSource_2[k,j,0]
                    Zero_2 = LeftShiftZeorSource_2[k,j,1].item()
                    originalResult = originalData_1 * originalData_2
                    if (Zero_1==1 or Zero_2==1):
                        SCResult = torch.tensor(0)
                        error = 0
                    else:
                        SCResult = BitstreamMUL(bitstream_1, bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog=math.log2(self.bitstreamlength), dataWidth=self.dataWidth).to(self.device)
                        error = abs(1 - SCResult.item() / originalResult.item())

                    result[i, j] += SCResult

        # for i in range(rows1):
        #     for j in range(cols2):
        #         for k in range(cols1):
        #             SCResult =  GenBitstream.SC_MUL(originData_1= self.tensor_1[i, k], originData_2 = self.tensor_2[k, j], rngSeq = self.rngSeq, dataWidth = self.dataWidth, device =self.device)
        #             result[i, j] += SCResult

        return result


if __name__ == "__main__":
    sobol_1 = [0, 16, 24, 8, 12, 28, 20, 4, 6, 22, 30, 14, 10, 26, 18, 2, 3, 19, 27, 11, 15, 31, 23, 7, 5, 21, 29, 13,
               9, 25, 17, 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sobolTensor = torch.tensor(sobol_1).to(device)
    tensor1 = torch.randint(0, 64, size=(256, 64))
    tensor2 = torch.randint(0, 64, size=(64, 8))

    GEMMKernel = SCbasedGEMM(tensor_1= tensor1 , tensor_2= tensor2,dataWidth= 8 ,rngSeq= sobolTensor ,device= device).to(device)
    GEMMResult = GEMMKernel()

    result = torch.matmul(tensor1, tensor2)
    print(result)