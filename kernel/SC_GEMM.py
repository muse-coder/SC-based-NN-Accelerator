import torch
from kernel.GenBitstream import  *
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

        self.rows1, self.cols1 = self.tensor_1.size()
        self.rows2, self.cols2 = self.tensor_2.size()
        # 执行矩阵乘法

        self.BitstreamSource_1 = torch.zeros(self.rows1, self.cols1, len(self.rngSeq)).to(self.device)
        self.BitstreamSource_2 = torch.zeros(self.rows2, self.cols2, len(self.rngSeq)).to(self.device)
        self.LeftShiftZeroSign_1 = torch.zeros(self.rows1, self.cols1, 3).to(self.device)
        self.LeftShiftZeroSign_2 = torch.zeros(self.rows2, self.cols2, 3).to(self.device)

        self.BitstreamSource_1, self.LeftShiftZeroSign_1 = self.GenBitstreamGEMM(self.rows1, self.cols1,
                                                                                 self.tensor_1, self.BitstreamSource_1,
                                                                                 self.LeftShiftZeroSign_1,self.rngSeq)

        self.BitstreamSource_2, self.LeftShiftZeroSign_2 = self.GenBitstreamGEMM(self.rows2, self.cols2,
                                                                                 self.tensor_2, self.BitstreamSource_2,
                                                                                 self.LeftShiftZeroSign_2,self.ascendingSeq)



    def GenBitstreamGEMM(self, row,col,tensor,BitstreamSource,LeftShiftZeroSign,rngSeq):
        for i in range(row):
            for j in range(col):
                if tensor[i, j] > 0:
                    sign = 1
                else:
                    sign = -1
                bitstream_1, leftShift_1, Zero_1 = GenBitstreamGroup(originData_1=abs(tensor[i, j]),
                                                                     rngSeq=rngSeq, dataWidth=self.dataWidth,
                                                                     device=self.device)
                originalData = abs(tensor[i, j])
                BitstreamSource[i, j, :] = bitstream_1
                LeftShiftZeroSign[i, j, 0] = leftShift_1
                LeftShiftZeroSign[i, j, 1] = Zero_1
                LeftShiftZeroSign[i, j, 2] = sign

                test_1 = bitstream_1.sum() * (2 ** (self.dataWidth-leftShift_1-math.log2(len(self.rngSeq))))

                # if (abs(1-(test_1 /originalData))>0.2):
                #     print("error")
                #
                # if sign * tensor[i, j] < 0 :
                #     print("error")

        return BitstreamSource , LeftShiftZeroSign

    def forward(self):

        approximateResult = torch.zeros(self.rows1, self.cols2).to(self.device)
        exactResult = torch.zeros(self.rows1, self.cols2).to(self.device)

        for i in range(self.rows1):
            for j in range(self.cols2):
                for k in range(self.cols1):
                    originalData_1 = self.tensor_1 [i,k]
                    bitstream_1 = self.BitstreamSource_1[i,k,:]
                    leftShift_1 = self.LeftShiftZeroSign_1[i,k,0]
                    Zero_1 = self.LeftShiftZeroSign_1[i,k,1].item()
                    sign_1 = self.LeftShiftZeroSign_1[i,k,2].item()

                    originalData_2 = self.tensor_2 [k,j]
                    bitstream_2 = self.BitstreamSource_2[k,j,:]
                    leftShift_2 = self.LeftShiftZeroSign_2[k,j,0]
                    Zero_2 = self.LeftShiftZeroSign_2[k,j,1].item()
                    sign_2 = self.LeftShiftZeroSign_2[k, j, 2].item()

                    originalResult = originalData_1 * originalData_2

                    if (Zero_1==1 or Zero_2==1):
                        signSCResult = torch.tensor(0)
                        error = 0
                    else:
                        absSCResult = BitstreamMUL(bitstream_1, bitstream_2,leftShift_1,leftShift_2,rngSeqLengthLog=math.log2(self.bitstreamlength), dataWidth=self.dataWidth).to(self.device)
                        signSCResult = absSCResult * sign_2 * sign_1
                        # error = abs(1 - signSCResult.item() / originalResult.item())
                        # if ((signSCResult * originalResult <0)):
                        #     print("error")

                        # if (error >0.1 ):
                        #     print(f"large error,i:{i},j:{j},k:{k},data_1:{originalData_1},data_2:{originalData_2}")


                    # if ((approximateResult[i, j]+signSCResult) / (exactResult[i, j]+originalResult) < 0):
                    #     print("big absulut error")

                    # if ( abs(1-(approximateResult[i, j]+signSCResult) / (exactResult[i, j]+originalResult)) >0.5):
                    #     print("big relative error")


                    approximateResult[i, j] += signSCResult
                    exactResult[i, j] += originalResult



        return approximateResult,exactResult


if __name__ == "__main__":
    sobol_1 = [0, 16, 24, 8, 12, 28, 20, 4, 6, 22, 30, 14, 10, 26, 18, 2, 3, 19, 27, 11, 15, 31, 23, 7, 5, 21, 29, 13,
               9, 25, 17, 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sobolTensor = torch.tensor(sobol_1).to(device)
    tensor1 = torch.randint(0, 128, size=(256, 64))
    tensor2 = torch.randint(0, 128, size=(64, 4))

    # tensor1 = torch.randint(-32, 32, size=(128, 32))
    # tensor2 = torch.randint(-32, 32, size=(32, 4))

    # random_tensor = torch.randint(min_value, max_value, (32, 64), dtype=torch.int32)

    GEMMKernel = SCbasedGEMM(tensor_1= tensor1 , tensor_2= tensor2,dataWidth= 8 ,rngSeq= sobolTensor ,device= device).to(device)
    GEMMResult,exactResult = GEMMKernel()

    result = torch.matmul(tensor1, tensor2).to(device)
    relativeError = abs(1-GEMMResult/result)
    relativeError_2 = abs(1-exactResult/result)
    # print(relativeError_2)
    maxError = torch.max(relativeError)
    minError = torch.min(relativeError)

    max_index = torch.argmax(relativeError)
    min_index = torch.argmin(relativeError)
    # 将一维索引转换为二维索引
    max_position = (max_index // relativeError.shape[1], max_index % relativeError.shape[1])
    min_position = (min_index // relativeError.shape[1], min_index % relativeError.shape[1])
    maxError = relativeError[max_position[0].item(), max_position[1].item()].item()
    minError = relativeError[min_position[0].item(), min_position[1].item()].item()
    print(maxError)
    # print(max_position)
    print(minError)
    # print(min_position)
