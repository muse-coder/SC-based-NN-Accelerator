# %%
import torch
from UnarySim.stream.gen import RNG, SourceGen, BSGen
from UnarySim.metric.metric import ProgError
import matplotlib.pyplot as plt
import time

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
bitwidth = 8
dim = 1
rng = "Sobol"
mode = "unipolar"
col = 100000

# %%
result_pe_cycle = []
if mode == "unipolar":
    iVec = ((torch.rand(1, col)-0.5)*2).mul(2**bitwidth).round().div(2**bitwidth).to(device)
elif mode == "bipolar":
    iVec = torch.rand(1, col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)

iVecSource = SourceGen(iVec, bitwidth=bitwidth, mode=mode)().to(device)

iVecRNG = RNG(bitwidth, dim, rng)().to(device)
iVecBS = BSGen(iVecSource, iVecRNG).to(device)

iVecPE = ProgError(iVec, mode=mode).to(device)
with torch.no_grad():
    idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
    start_time = time.time()
    for i in range(2**bitwidth):
        # if (i == 2**bitwidth-3) :
        #     print("ending bitstream generation")
        #     print(i)
        iBS = iVecBS(idx + i)
        iVecPE.Monitor(iBS)
        result_pe_cycle.append(1-torch.sqrt(torch.sum(torch.mul(iVecPE()[1][0], iVecPE()[1][0]))/col).item())#iVecPE()[1][0] 是什么意思？
    print("--- %s seconds ---" % (time.time() - start_time))
    print("input error: ", "min:", torch.min(iVecPE()[1]).item(), "max:", torch.max(iVecPE()[1]).item())
    result_pe = iVecPE()[1][0].cpu().numpy()
    print("error distribution=========>")
    plt.figure(figsize=(3,1.5))
    fig = plt.hist(result_pe, bins='auto')  # arguments are passed to np.histogram
    plt.title("data: "+mode)
    plt.show()
    print("progressive accuracy=========>")
    plt.figure(figsize=(3,1.5))
    fig = plt.plot(result_pe_cycle)  # arguments are passed to np.histogram
    plt.title("data: "+mode)
    plt.show()

# %%
