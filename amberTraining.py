import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import grad

Mask, IDs, Seq, PSSM, ACC, Coords, SS, ASA, Seq_PDB, Coords_PDB, AmEn, AmEnSide, AmEnBB, AmEtotal = torch.load('AT_Energy_Array_100.pt')

def energyPerRes(E,S, allEs=[torch.zeros(1)] * 20):
    n = E.numel()
    for i in range(n):
        #print(S[i]-1)
        allEs[S[i]-1] = torch.cat((allEs[S[i]-1], E[i].unsqueeze(0)))

    return allEs

allEs = energyPerRes(torch.tensor(AmEn[0]),torch.tensor(Seq[0]))
for i in range(100):
    ei = torch.tensor(AmEn[i])
    if ei.max() < 200:
       allEs = energyPerRes(torch.tensor(AmEn[i]),torch.tensor(Seq[i]))

plt.figure(1)
i = 0
for j in range(4):
    for k in range(5):
        #h, n = torch.histogram(allEs[i], bins=25)
        #h = h / h.sum()
        #nmid = (n[:-1]+n[1:])/2
        plt.subplot(4,5,i+1)
        #plt.plot(nmid,h)

        counts, bins = np.histogram(allEs[i].numpy(), bins=20)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.title(i+1)
        i += 1





