#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.rnn import HUBMGUCell, HardMGUCell, HardMGUCellFxp
from UnarySim.stream.gen import *
from UnarySim.metric.metric import SourceGen, RNG, BSGen, ProgError
from UnarySim.kernel.utils import truncated_normal, progerror_report
import time, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bitwidth_list = [4, 5, 6, 7, 8, 9, 10, 11, 12]
output_dir = "/home/diwu/Project/UnarySim/app/uBrain/layer_eval/"

win_sz = 10 # win size
batch = 32
input_sz = 256 # fc size
hidden_sz = 64 # hidden size
intwidth = 1 # systolic array fxp
mode = "bipolar"
rng = "Sobol"
bias = False

err_array = np.zeros((len(bitwidth_list), 3, win_sz))
std_array = np.zeros((len(bitwidth_list), 3, win_sz))

outfile = "layer_eval_mgu_log.csv"
fp = open(output_dir+outfile, "w")

out_err_array = output_dir + "/layer_eval_mgu_err_array.npy"
out_std_array = output_dir + "/layer_eval_mgu_std_array.npy"

if not (os.path.exists(out_err_array) and os.path.exists(out_std_array)):
    for bitwidth_index in range(len(bitwidth_list)):
        bitwidth = bitwidth_list[bitwidth_index]
        print("bit width:", bitwidth)

        fracwidth = bitwidth - intwidth
        depth = bitwidth + 4
        if bitwidth <= 11:
            depth_ismul = 6
        else:
            depth_ismul = 7

        input = torch.randn(win_sz, batch, input_sz).to(device)
        input = truncated_normal(input, mean=0, std=0.4)
        hx1 = torch.randn(batch, hidden_sz).to(device)
        hx1 = truncated_normal(hx1, mean=0, std=0.1)
        hx2 = hx1.clone().detach().to(device)
        hx3 = hx1.clone().detach().to(device)
        hx4 = hx1.clone().detach().to(device)
        output1 = []
        output2 = []
        output3 = []
        output4 = []

        rnn1 = HardMGUCell(input_sz, hidden_sz, bias=bias, hard=True).to(device)

        rnn2 = HUBMGUCell(input_sz, hidden_sz, bias=bias, 
                        binary_weight_f=rnn1.weight_f, binary_bias_f=rnn1.bias_f, binary_weight_n=rnn1.weight_n, binary_bias_n=rnn1.bias_n, 
                        rng=rng, bitwidth=bitwidth, mode=mode, depth=depth, depth_ismul=depth_ismul).to(device)

        rnn3 = HardMGUCellFxp(input_sz, hidden_sz, bias=bias, hard=True, intwidth=intwidth, fracwidth=fracwidth).to(device)
        rnn3.weight_f.data = rnn1.weight_f.clone().detach().to(device)
        rnn3.weight_n.data = rnn1.weight_n.clone().detach().to(device)

        rnn4 = HUBMGUCell(input_sz, hidden_sz, bias=bias, 
                        binary_weight_f=rnn1.weight_f, binary_bias_f=rnn1.bias_f, binary_weight_n=rnn1.weight_n, binary_bias_n=rnn1.bias_n, 
                        rng=rng, bitwidth=bitwidth+1, mode=mode, depth=depth, depth_ismul=depth_ismul).to(device)

        for i in range(win_sz):
            hx1 = rnn1(input[i], hx1)
            output1.append(hx1)

            hx2 = rnn2(input[i], hx2)
            output2.append(hx2)

            hx3 = rnn3(input[i], hx3)
            output3.append(hx3)

            hx4 = rnn4(input[i], hx4)
            output3.append(hx4)

            hub_err = hx1 - hx2
            min = hub_err.min().item()
            max = hub_err.max().item()
            rmse = torch.sqrt(torch.mean(torch.square(hub_err)))
            std, mean = torch.std_mean(hub_err)
            log = "{:30s}".format(str(i)+"-th win output hub") + \
                    ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                    ", std," + "{:12f}".format(std) + \
                    ", mean," + "{:12f}".format(mean) + \
                    ", rmse," + "{:12f}".format(rmse)
            print(log)
            fp.write(log+"\n")
            err_array[bitwidth_index, 0, i] = rmse.cpu().item()
            std_array[bitwidth_index, 0, i] = std.cpu().item()

            fxp_err = hx1 - hx3
            min = fxp_err.min().item()
            max = fxp_err.max().item()
            rmse = torch.sqrt(torch.mean(torch.square(fxp_err)))
            std, mean = torch.std_mean(fxp_err)
            log = "{:30s}".format(str(i)+"-th win output fxp") + \
                    ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                    ", std," + "{:12f}".format(std) + \
                    ", mean," + "{:12f}".format(mean) + \
                    ", rmse," + "{:12f}".format(rmse)
            print(log)
            fp.write(log+"\n")
            err_array[bitwidth_index, 1, i] = rmse.cpu().item()
            std_array[bitwidth_index, 1, i] = std.cpu().item()

            sc_err = hx1 - hx4
            min = sc_err.min().item()
            max = sc_err.max().item()
            rmse = torch.sqrt(torch.mean(torch.square(sc_err)))
            std, mean = torch.std_mean(sc_err)
            log = "{:30s}".format(str(i)+"-th win output sc") + \
                    ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                    ", std," + "{:12f}".format(std) + \
                    ", mean," + "{:12f}".format(mean) + \
                    ", rmse," + "{:12f}".format(rmse)
            print(log)
            fp.write(log+"\n")
            err_array[bitwidth_index, 2, i] = rmse.cpu().item()
            std_array[bitwidth_index, 2, i] = std.cpu().item()
        print()
    fp.close()

    np.save(out_err_array, err_array)
    np.save(out_std_array, std_array)

err_array = np.load(out_err_array)
std_array = np.load(out_std_array)

print(err_array)
print(std_array)

font = {'family':'Times New Roman', 'size': 6}
matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 0.8
fig_w = 3.6
alpha = 1

labels = [str(bitwidth) for bitwidth in bitwidth_list]
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=my_dpi)

for idx in range(len(bitwidth_list)):
    data_hub = err_array[idx, 0, :]
    data_fxp = err_array[idx, 1, :]
    data_sc  = err_array[idx, 2, :]
    std_hub  = std_array[idx, 0, :]
    std_fxp  = std_array[idx, 1, :]
    std_sc   = std_array[idx, 2, :]
    interval = 1/2**(win_sz.bit_length())
    x_axe = [(x[idx] - (win_sz - 1) / 2 * interval + x_tick * interval) for x_tick in range(win_sz)]
    if idx == 0:
        ax.plot(x_axe, data_fxp, "-^", label="Systolic", alpha=alpha, color="#7A81FF", lw=0.5, ms=0.5)
        ax.plot(x_axe, data_sc, "-P", label="SC", alpha=alpha, color="#AAAAAA", lw=0.5, ms=0.5)
        ax.plot(x_axe, data_hub, "-s", label="uBrain", alpha=alpha, color="#FF7F7F", lw=0.5, ms=0.5)
    else:
        ax.plot(x_axe, data_fxp, "-^", alpha=alpha, color="#7A81FF", lw=0.5, ms=0.5)
        ax.plot(x_axe, data_sc, "-P", alpha=alpha, color="#AAAAAA", lw=0.5, ms=0.5)
        ax.plot(x_axe, data_hub, "-s", alpha=alpha, color="#FF7F7F", lw=0.5, ms=0.5)

    ax.fill_between(x_axe, data_fxp + std_fxp / 2, data_fxp - std_fxp / 2, alpha=0.3, color="#7A81FF", lw=0.0)
    ax.fill_between(x_axe, data_sc  + std_sc  / 2, data_sc  - std_sc  / 2, alpha=0.3, color="#AAAAAA", lw=0.0)
    ax.fill_between(x_axe, data_hub + std_hub / 2, data_hub - std_hub / 2, alpha=0.3, color="#FF7F7F", lw=0.0)


locs = [0, 0.2, 0.4, 0.6, 0.8]
ax.set_yticks(locs)
ax.set_yticklabels(locs)


ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('RMSE-STD\n')
ax.legend(ncol=3, frameon=True)
fig.tight_layout()
fig.savefig(output_dir+"layer_eval_mgu.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
