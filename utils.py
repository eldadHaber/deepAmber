import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

import time
#from src \
import graphOps as GO
#from src \
import constraints as cons


def lossFundRMSD(P, Q, M, contact=1e6):
    ind = M.squeeze() > 0
    P = P[:, :, ind].squeeze(0)
    Q = Q[:, :, ind].squeeze(0)

    p = torch.sum(P ** 2, dim=0, keepdim=True)
    q = torch.sum(Q ** 2, dim=0, keepdim=True)

    DP = torch.triu(torch.sqrt(torch.relu(p + p.t() - 2 * P.t() @ P)), 1)
    DQ = torch.triu(torch.sqrt(torch.relu(q + q.t() - 2 * Q.t() @ Q)), 1)

    con = (DQ < contact)

    DP = DP[con]
    DQ = DQ[con]

    a = (DQ > 0)
    DP = DP[a]
    DQ = DQ[a]

    return F.mse_loss(DP, DQ)  #/F.mse_loss(DQ*0,DQ)



def lossFunNG(P, Q, M, dim=3):
    ind = M.squeeze() > 0
    P = P[:, :, ind].squeeze(0)
    Q = Q[:, :, ind].squeeze(0)
    P = P.reshape((dim, 3, -1))
    Q = Q.reshape((dim, 3, -1))

    Pmean = P.mean(dim=[0, 2], keepdim=True)
    Qmean = Q.mean(dim=[0, 2], keepdim=True)

    # go to the center
    P = P - Pmean
    Q = Q - Qmean

    Q = Q.transpose(1, 0)
    P = P.transpose(1, 0)
    Q = Q.reshape((3, -1))
    P = P.reshape((3, -1))

    with torch.no_grad():
        # Rotate P->Q
        H = P @ Q.t()
        # print(H)
        U, S, V = torch.svd(H)
        RPQ = V @ U.t()
        RQP = U @ V.t()

    loss = 0.5 * (F.mse_loss(RPQ @ P, Q) + F.mse_loss(RQP @ Q, P))
    # loss = 0.5*(F.l1_loss(RPQ @ P, Q) + F.l1_loss(RQP @ Q, P))

    return loss


def quadu(x):
    y = torch.zeros_like(x)
    ind = x > 0
    y[ind] = 0.5 * (x[ind] ** 2)
    return y


def intTanh(x):
    # s always has real part >= 0
    s = torch.sign(x) * x
    p = torch.exp(-2 * s)
    return s + torch.log1p(p) - np.log(2)


def getTrainingData(coordAlpha, coordC, coordN, seq, pssm, msk, i, device='cpu'):
    scale = 1e-2
    PSSM = pssm[i].t()
    n = PSSM.shape[0]
    M = msk[i][:n]
    A = seq[i].t()

    X1 = coordAlpha[i].t()
    X2 = coordC[i].t()
    X3 = coordN[i].t()

    Coords = scale * torch.cat((X1, X2, X3), dim=1)

    Coords = Coords.type('torch.FloatTensor')
    Coords = Coords.to(device=device, non_blocking=True)

    PSSM = PSSM.type(torch.float32)

    # nodalFeat = torch.cat((PSSM, A, entropy[i].unsqueeze(1)),dim=1)
    nodalFeat = torch.cat((PSSM, A), dim=1)

    nodalFeat = nodalFeat.to(torch.float32)
    nodalFeat = nodalFeat.to(device=device, non_blocking=True)

    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)

    # Find neibours
    D = torch.relu(torch.sum(Coords ** 2, dim=1, keepdim=True) + \
                   torch.sum(Coords ** 2, dim=1, keepdim=True).t() - \
                   2 * Coords @ Coords.t())

    # D = torch.eye(D.shape[0])
    neibors = 15
    ndiags = np.min([neibors, D.shape[0]])
    D  = torch.triu(D, 1)
    Di = torch.diag(torch.diag(D, 1), 1)
    D  = D - Di

    for k in range(2, ndiags):
        Di = torch.diag(torch.diag(D, k), k)
        ei = torch.ones(D.shape[0] - k, device=device)
        Ei = torch.diag(ei, k)
        D = D + Ei - Di

    S = (D > 0) * (D < 3*(12 ** 2))
    I, J = torch.nonzero(S, as_tuple=True)

    I = I.to(device=device, non_blocking=True)
    J = J.to(device=device, non_blocking=True)

    return nodalFeat.t().unsqueeze(0), Coords.t().unsqueeze(0), M.unsqueeze(0).unsqueeze(0), I, J

def getBatchTrainingData(coordAlpha, coordBeta, coordN, seq, pssm, msk, I, device='cpu'):

    cnt = 0
    I = []; J = []; M = []; Coords = []; nodalFeat = []; batchVec = []
    j = 0
    for i in I:
        nFi, Coordsi, Mi, Ii, Ji = \
            getTrainingData(coordAlpha, coordBeta, coordN, seq, pssm, msk, i, device)
        n = nFi.shape[-1]
        I.append(Ii + cnt)
        J.append(Ji + cnt)
        M.append(Mi)
        Coords.append(Coordsi)
        nodalFeat.append(nFi)
        batchVec.append(torch.ones(n)*j)
        j += 1
        cnt += n

    I = torch.cat(I)
    J = torch.cat(J)
    M = torch.cat(M, dim=2)
    Coords = torch.cat(Coords, dim=2)
    nodalFeat = torch.cat(nodalFeat, dim=2)
    batchVec  = torch.cat(batchVec)

    return nodalFeat, Coords, M, I, J, batchVec

def recomputeGraph(xnS, Kclose, Graph, ns=33):
    device = xnS.device
    with torch.no_grad():
        # Initialize graph
        Iind = Graph.iInd.to(device)
        Jind = Graph.jInd.to(device)

        nodes = xnS.shape[-1]
        D0 = torch.zeros(nodes, nodes, device=device)
        D0[Iind, Jind] = 1.0

        # Compute new edges
        if len(xnS.shape) == 3:  # Scalar input
            Coords = F.conv1d(xnS, Kclose.unsqueeze(2))
        else:  # Vector input
            Coords = F.conv2d(xnS, Kclose)

        Id = torch.ones(1, 1, 1, 1) * 1.0
        Coords = cons.proj(Coords, Id, iter=100)

        Coords = Coords.squeeze()
        D = torch.relu(torch.sum(Coords ** 2, dim=0, keepdim=True) + \
                       torch.sum(Coords ** 2, dim=0, keepdim=True).t() - \
                       2 * Coords.t() @ Coords)
        D = D / D.std()
        D = torch.exp(-2 * D) + D0
        D = torch.triu(D, 1)
        nsparse = ns
        vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
        nd = D.shape[0]
        k = min(nsparse, D.shape[0])
        I = torch.ger(torch.arange(nd, dtype=torch.float32).to(device),
                      torch.ones(k, dtype=torch.float32, device=device))
        I = I.view(-1).type(torch.int64)
        J = indices.view(-1).type(torch.int64)

        ind = (J - I) > 0

        Graph.iInd = I[ind]
        Graph.jInd = J[ind]

    return Graph


def colissionPenalty(x, r):
    i = x < r
    y = torch.zeros_like(x)
    y[i] = torch.log(x[i] / r + 1e-8) - x[i] / r + 1.0

    return -y


def addCbeta(X):
    A = X[:, :, :3, :]
    C = X[:, :, 3:6, :]
    N = X[:, :, 6:, :]

    NA = N - A
    CA = C - A
    normal = torch.zeros_like(A)
    normal[:, :, 0, :] = NA[:, :, 2, :] * CA[:, :, 1, :] - NA[:, :, 1, :] * CA[:, :, 2, :]
    normal[:, :, 1, :] = NA[:, :, 0, :] * CA[:, :, 2, :] - NA[:, :, 2, :] * CA[:, :, 0, :]
    normal[:, :, 2, :] = NA[:, :, 1, :] * CA[:, :, 0, :] - NA[:, :, 0, :] * CA[:, :, 1, :]

    normal = normal / torch.sqrt(torch.sum(normal ** 2, dim=2, keepdim=True) + 1e-3)

    normal = normal + A

    Xout = torch.cat([A, C, N, normal], dim=2)

    return Xout


def tetraDis(Tet, TetTar, M):
    # loss between Tetras
    n = Tet.shape
    nnodes = n[-1]
    iInd = torch.arange(nnodes - 1)
    jInd = iInd + 1

    T = Tet.reshape(n[0], 4, 3, nnodes)
    Ttar = TetTar.reshape(n[0], 4, 3, nnodes)

    I = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=T.device)
    J = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], device=T.device)

    TI = T[:, I, :, :]
    TJ = T[:, J, :, :]
    G = TI[:, :, :, iInd] - TJ[:, :, :, jInd]
    d = torch.sqrt(torch.sum(G ** 2, dim=2))

    TI = Ttar[:, I, :, :]
    TJ = Ttar[:, J, :, :]
    G = TI[:, :, :, iInd] - TJ[:, :, :, jInd]
    dtar = torch.sqrt(torch.sum(G ** 2, dim=2))

    Me = 0.5 * (M[:, :, :-1] + M[:, :, 1:])
    Me = (Me == 1)

    lossTT = F.mse_loss(Me * d, Me * dtar) # / F.mse_loss(torch.zeros_like(dtar), Me * dtar)

    # loss inside a Tetra
    I = torch.tensor([0, 0, 0, 1, 1, 2], device=T.device)
    J = torch.tensor([1, 2, 3, 2, 3, 3], device=T.device)
    G = T[:, I, :, :] - T[:, J, :, :]
    d = torch.sqrt(torch.sum(G ** 2, dim=2))

    G = Ttar[:, I, :, :] - Ttar[:, J, :, :]
    dtar = torch.sqrt(torch.sum(G ** 2, dim=2))

    lossTI = F.mse_loss(M * d, M * dtar) #/ F.mse_loss(torch.zeros_like(dtar), M * dtar)

    return lossTT + lossTI