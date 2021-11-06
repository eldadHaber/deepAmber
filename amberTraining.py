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
#Mask, IDs, Seq, PSSM, ACC, Coords, SS, ASA, Seq_PDB, Coords_PDB, AmEn, AmEnSide, AmEnBB, AmEtotal = torch.load('AT_Energy_Array_100.pt')
#Mask, IDs, Seq, PSSM, ACC, Coords, SS, ASA, Seq_PDB, Coords_PDB, AmEn, AmEnSide, AmEnBB, AmEtotal = torch.load('AT_Energy_Array_100_cutoff.pt')

IDs, Seq, PSSM, Seq_PDB, Coords_PDB, AmEn, AmEnSide, AmEnBB, AmEtotal = torch.load('AT_E_PSSM.pt')

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
        #if ei.max() < 200:
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


def simplexGrad(enet,xnS, X3, Graph, h=0.1, k=16):
    dE = torch.zeros(k)
    V  = torch.zeros(k, X3.numel())

    for i in range(k):
        dX = torch.sign(torch.randn_like(X3))
        EP = enet(xnS, X3+h*dX, Graph)
        V[i,:] = dX.view(-1)
        dE[i] = EP.sum()

    A = torch.cat((torch.ones(k, 1), h*V), dim=1)
    gradX = torch.linalg.solve(A@A.T + 1e-3*torch.eye(k), dE)

    return (A.T@gradX)[1:]


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

ndata = 100
epochs = 100


for i in range(epochs):
    torch.cuda.empty_cache()
    for j in range(ndata):
        # ========= Prepare the data
        C = Coords_PDB[j]
        S = torch.tensor(Seq[j])
        I = torch.arange(len(S));
        seq = torch.zeros(20,len(S))
        seq[S-1, I] = 1
        seq = seq.unsqueeze(0)
        #coordAlpha = torch.tensor(C[:,:,0]).unsqueeze(0)
        #coordBeta  = torch.tensor(C[:,:,1]).unsqueeze(0)
        #coordN     = torch.tensor(C[:,:,2]).unsqueeze(0)
        pssm = torch.tensor(PSSM[j]).unsqueeze(0)
        msk = torch.ones(1,pssm.shape[-1])
        coordAlpha = torch.tensor(C[::3, :]).T.unsqueeze(0)
        coordBeta = torch.tensor(C[1::3, :]).T.unsqueeze(0)
        coordN = torch.tensor(C[2::3, :]).T.unsqueeze(0)

        xnS, X, M, I, J = utils.getTrainingData(coordAlpha, coordBeta, coordN,
                                                     seq, pssm, msk, 0, device=device)
        CoordsBeta = utils.addCbeta(X.unsqueeze(1))
        CoordsBeta, res = cons.proj(CoordsBeta, iter=1000, tol=1e-2)

        Ej = torch.tensor(AmEn[j])
        # ==========================
        nodes = xnS.shape[2]
        if nodes > 400:
            continue

        # Generator
        Graph = GO.vectorGraph(I, J, nodes)
        X3 = CoordsBeta.clone()
        X3.requires_grad = True
        optimizer.zero_grad()
        Ecomp = enet(xnS, X3, Graph)
        # Collapse the energy to a single channel
        #grade = grad(Ecomp.sum(), X3, retain_graph=True)[0]
        #grade = simplexGrad(enet, xnS, X3, Graph, h=0.1, k=10)
        lossGrad = torch.zeros(1) #F.mse_loss(grade, torch.zeros_like(grade))
        loss = ((Ecomp-Ej)**2).sum()/(Ej**2).sum() + lossGrad
        loss.backward()
        torch.nn.utils.clip_grad_value_(enet.parameters(), 0.3)
        optimizer.step()

        print('Iter =  %2d   %2d   loss = %3.2e   lossGrad = %3.2e'%(i, j, loss.item(), lossGrad.item()))


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
    Xout, Eout = eNet.updateCoords(enet, X3, xnS, Graph, lrX=1e-1, iter=500, Emin=0.0, ratio=0.01)
    dRMSDout = utils.lossFundRMSD(Xout.squeeze(0), CoordsBeta.squeeze(0), M, contact=1e3)



