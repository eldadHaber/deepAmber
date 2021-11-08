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
import energyNetV5 as eNet
#from src \
import constraints as cons
import sys
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# sample = True
base_path = '../SampleData'
# base_path = '../data/casp11'
# load training data
coordN = torch.load(base_path + '/CoordN.pt')
coordAlpha = torch.load(base_path + '/CoordAlpha.pt')
coordBeta = torch.load(base_path + '/CoordBeta.pt')

pssm = torch.load(base_path + '/pssm.pt')
entropy = torch.load(base_path + '/entropy.pt')
seq = torch.load(base_path + '/seq.pt')
msk = torch.load(base_path + '/mask.pt')

#RamaPoints = torch.load(base_path + '/RamaPoints.pt')
#DistPoints = torch.load(base_path + '/DistPoints.pt')

# ============================================================
#

nlayer = 6
nopen = 32
enet = eNet.energyGraphNetwork(nopen, nlayer)

enet.to(device)
params = sum(p.numel() for p in enet.parameters())

print('Number of parameters ', params)

# f(x) = x^2 + 0.01* sin(100000*x)    f' = 2*x * 1e3*cos(100000*x)
# f(X + h*V) = f(X) + h*V'*\grad f + HOT
#  [1          ] [f(x)   ]     [f(X+h*v1]
#  [0    v1^T  ] [\grad f]   = [  ..]
#  [0    v2^T  ]
#  [0   ...    ]
#  [0    v_n^T ]               [f(X+h*vn)]
#
def simplexGrad(enet,meanEnergyPerRes,xnS, X3, Graph, h=0.1, k=16):
    dE = torch.zeros(k)
    V  = torch.zeros(k, X3.numel())

    for i in range(k):
        dX = torch.sign(torch.randn_like(X3))
        EP = enet(xnS, X3+h*dX, Graph)
        resType = xnS[:, 20:, :].squeeze().T
        mu = resType@meanEnergyPerRes

        V[i,:] = dX.view(-1)
        dE[i] = (EP).mean()

    A = torch.cat((torch.ones(k, 1), h * V), dim=1)
    gradX = torch.linalg.solve(A @ A.T + 1e-3 * torch.eye(k), dE)

    return (A.T @ gradX)[1:]


lr = 3e-4

meanEnergyPerRes = nn.Parameter(torch.zeros(20))
optimizer = optim.Adam([{'params': enet.parameters(), 'lr': lr},{'params': meanEnergyPerRes, 'lr': lr}])

ndata = 64
epochs = 200


for i in range(epochs):
    torch.cuda.empty_cache()
    for j in range(ndata):
        # ========= Prepare the data
        xnS, Coords, M, I, J = utils.getTrainingData(coordAlpha, coordBeta, coordN,
                                                     seq, pssm, msk, j, device=device)
        CoordsBeta = utils.addCbeta(Coords.unsqueeze(1))
        CoordsBeta, res = cons.proj(CoordsBeta, iter=1000, tol=1e-2)
        # ==========================
        nodes = xnS.shape[2]
        if nodes > 400:
            continue

        # Generator
        Graph = GO.vectorGraph(I, J, nodes)
        X3 = CoordsBeta

        optimizer.zero_grad()
        e0 = enet(xnS, X3, Graph)
        # Collapse the energy to a single channel
        resType = xnS[:, 20:, :].squeeze().T
        mu = resType@meanEnergyPerRes

        meanEcomp = (e0-mu).mean()
        muSq = (e0-mu)**2
        sigma = e0.std()
        lossMLE = 0.5*(muSq.mean() + sigma**2 - torch.log(sigma) - 1)

        gradX3 = simplexGrad(enet, meanEnergyPerRes, xnS, X3, Graph, h=0.1, k=16)
        lossE = F.mse_loss(gradX3, torch.zeros_like(gradX3))

        loss = lossE + lossMLE
        loss.backward()
        torch.nn.utils.clip_grad_value_(enet.parameters(), 0.1)
        optimizer.step()

        print('Iter =  %2d   %2d   loss = %3.2e   lossMLE = %3.2e   lossE = %3.2e   mu = %3.2e   sigma = %3.2e    ProtEnergy = %3.2e'%
              (i, j, loss.item(), lossMLE.item(),lossE.item(), (muSq).mean().item(), sigma.item(), meanEcomp.item()))


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

