import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import grad

import time
import graphOps as GO
#from src
import utils as utils
#from src
import energyNetwork as eNet
#from src \
import constraints as cons

# load the data and plot it
Mask, IDs, Seq, PSSM, ACC, Coords, SS, ASA, Seq_PDB, Coords_PDB, AmEn, AmEnSide, AmEnBB, AmEtotal = torch.load('AT_Energy_Array_100.pt')

pltStat = False
if pltStat:

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
            plt.subplot(4,5,i+1)

            counts, bins = np.histogram(allEs[i].numpy(), bins=20)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.title(i+1)
            i += 1



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ============================================================
#

nlayer = 6
nopen = 32
enet = eNet.energyGraphNetwork(nopen, nlayer)

enet.to(device)
params = sum(p.numel() for p in enet.parameters())

print('Number of parameters ', params)

lr = 1e-2

optimizer = optim.Adam([{'params': enet.parameters(), 'lr': lr}])

ndata = 1
epochs = 100


for i in range(epochs):
    torch.cuda.empty_cache()
    for j in range(ndata):
        # ========= Prepare the data
        C = Coords[j]
        S = Seq[j]; I = torch.arange(len(S)); seq = torch.zeros(20,len(S))
        seq[S-1, I] = 1
        seq = seq.unsqueeze(0)
        coordAlpha = torch.tensor(C[:,:,0]).unsqueeze(0)
        coordBeta  = torch.tensor(C[:,:,1]).unsqueeze(0)
        coordN     = torch.tensor(C[:,:,2]).unsqueeze(0)
        pssm = torch.tensor(PSSM[j]).unsqueeze(0)
        msk = torch.tensor(Mask[j]).unsqueeze(0)

        xnS, X, M, I, J = utils.getTrainingData(coordAlpha, coordBeta, coordN,
                                                     seq, pssm, msk, j, device=device)
        CoordsBeta = utils.addCbeta(X.unsqueeze(1))
        CoordsBeta, res = cons.proj(CoordsBeta, iter=1000, tol=1e-2)

        Ej = torch.tensor(AmEn[j])
        # ==========================
        nodes = xnS.shape[2]
        if nodes > 400:
            continue

        # Generator
        Graph = GO.vectorGraph(I, J, nodes)
        X3 = CoordsBeta

        optimizer.zero_grad()
        Ecomp = enet(xnS, X3, Graph)
        # Collapse the energy to a single channel
        # loss = F.mse_loss(Ecomp1D.squeeze(),Ej)
        loss = ((Ecomp-Ej)**2).sum()/(Ej**2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_value_(enet.parameters(), 0.3)
        optimizer.step()

        print('Iter =  %2d   %2d   loss = %3.2e'%(i, j, loss.item()))


test = False
if test:
    R = 3.0
    dX = torch.randn_like(CoordsBeta)
    CoordsBetaPer = CoordsBeta + R * dX
    CoordsBetaPer, res = cons.proj(CoordsBetaPer, iter=1000, tol=1e-2)
    # ==========================
    nodes = xnS.shape[2]

    # Generator
    Graph = GO.vectorGraph(I, J, nodes)

    X3 = CoordsBetaPer
    dRMSDin = utils.lossFundRMSD(X3.squeeze(0), CoordsBeta.squeeze(0), M, contact=1e3)
    Xout, Eout = eNet.updateCoords(enet, X3, xnS, Graph, lrX=1e-1, iter=100, Emin=0.0, ratio=0.01)
    dRMSDout = utils.lossFundRMSD(Xout.squeeze(0), CoordsBeta.squeeze(0), M, contact=1e3)



