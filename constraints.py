import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


# Without collision
# min \|X - Xobs\|^2   s.t c(X) = 0  and collisions
#  X - Xobs  - G'*lam = 0
#  C(X) = 0
#
#  1. X = Xobs + G'*lam
#  2. C(Xobs + G'*lam) = 0
#
#  C(Xobs) + G*G'*lam = 0

# Without collision
# min \|X - Xobs\|^2 - mu*log(d(X))   s.t c(X) = 0  and collisions
#  X - Xobs - mu* D'*1/d(X) - G'*lam = 0
#  C(X) = 0
#
#  1. X = Xk + G'*lam + mu* D'*1/d(Xk)
#  2. C(Xk + G'*lam + mu * D'*1/d(Xk)) = 0
#
#  C(Xk + mu * D'*1/d(Xk)) + G*G'*lam = 0



import time
#from src \
import graphOps as GO

# Deal with constaints
def diffX(X):
    #X = X.squeeze()
    return X[:,:,:,1:] - X[:,:,:,-1]

def diffXT(X):
    #X  = X.squeeze()
    D  = X[:,:,:,:-1] - X[:,:,:,1:]
    d0 = -X[:,:,:,0].unsqueeze(3)
    d1 = X[:,:,:,-1].unsqueeze(3)
    D  = torch.cat([d0,D,d1],dim=3)
    return D


def disLin(X,Y):
    return torch.sum((X - Y)**2, dim=2, keepdim=True)

def dDisLinMV(S, X,Y):
    # (X-Y + S)**2
    R = X-Y
    G = 2*(R*S).sum(dim=2,keepdim=True)
    return G

def dDisLinMVT(w, X,Y):
    R = X-Y
    W = torch.repeat_interleave(w,3,dim=2)
    V = 2*R*W

    return V


def constraints(X):
    # define distances(sq) on the triangle
    rAC = 1.523**2
    rNC = 2.45**2  # NOT SURE ...
    rNA = 1.457**2
    rBA = 1.0**2
    rBN = 1.767**2
    rBC = 1.822**2

    # define distances(sq) on sequencial atoms
    rANp = 2.43**2
    rAAp = 3.79**2
    rCNp = 1.33**2
    rCAp = 2.43**2


    A = X[:,:,:3,:]
    C = X[:,:,3:6,:]
    N = X[:,:,6:9,:]
    B = X[:,:,9:,:]

    # Trtra constraints
    cAC   = disLin(A,C) - rAC
    cNC   = disLin(N,C) - rNC
    cNA   = disLin(N,A) - rNA
    cBC   = disLin(B,C) - rBC
    cBA   = disLin(B,A) - rBA
    cBN   = disLin(B,N) - rBN


    # alpha - alpha+
    cAAp = disLin(A[:,:,:,1:],A[:,:,:,:-1]) - rAAp
    # C - alpha+
    cCAp = disLin(A[:,:,:, 1:], C[:,:,:, :-1]) - rCAp
    # alpha - N+
    cANp = disLin(N[:,:,:, 1:], A[:,:,:, :-1]) - rANp
    # C - N+
    cCNp = disLin(N[:,:,:, 1:], C[:,:,:, :-1]) - rCNp

    return cAC, cNC, cNA, cBA, cBC, cBN, cAAp, cCAp, cANp, cCNp

def dConstraintsMV(dX, X):
    A = X[:,:,:3,:]
    C = X[:,:,3:6,:]
    N = X[:,:,6:9,:]
    B = X[:,:,9:, :]

    dA = dX[:,:,:3,:]
    dC = dX[:,:,3:6,:]
    dN = dX[:,:,6:9,:]
    dB = dX[:,:,9:, :]

    dcAC   = dDisLinMV(dA-dC, A,C)
    dcNC   = dDisLinMV(dN-dC, N,C)
    dcNA   = dDisLinMV(dN-dA, N,A)
    dcBA   = dDisLinMV(dB-dA, B,A)
    dcBC   = dDisLinMV(dB-dC, B,C)
    dcBN   = dDisLinMV(dB-dN, B,N)

    # alpha - alpha+
    dcAAp = dDisLinMV(dA[:,:,:,1:]-dA[:,:,:,:-1], A[:,:,:,1:],A[:,:,:,:-1])
    # C - alpha+
    dcCAp = dDisLinMV(dA[:,:,:, 1:]-dC[:,:,:, :-1], A[:,:,:, 1:], C[:,:,:,:-1])
    # A - N+
    dcANp = dDisLinMV(dN[:,:,:, 1:]-dA[:,:,:, :-1], N[:,:,:, 1:], A[:,:,:, :-1])
    # C - N+
    dcCNp = dDisLinMV(dN[:,:,:, 1:]-dC[:,:,:, :-1], N[:,:,:, 1:], C[:,:,:, :-1])

    return dcAC, dcNC, dcNA, dcBA, dcBC, dcBN, dcAAp, dcCAp, dcANp, dcCNp




def dConstraintsTMV(dcAC, dcNC, dcNA, dcBA, dcBC, dcBN, dcAAp, dcCAp, dcANp, dcCNp, X):

    A = X[:,:,:3,:]
    C = X[:,:,3:6,:]
    N = X[:,:,6:9,:]
    B = X[:,:,9:,:]

    dA = torch.zeros_like(A)
    dC = torch.zeros_like(C)
    dN = torch.zeros_like(N)
    dB = torch.zeros_like(B)

    dAdC   = dDisLinMVT(dcAC, A,C)
    dA = dA + dAdC
    dC = dC-dAdC
    dNdC   = dDisLinMVT(dcNC, N,C)
    dN = dN + dNdC
    dC = dC-dNdC
    dNdA = dDisLinMVT(dcNA, N,A)
    dN = dN + dNdA
    dA = dA - dNdA

    dBdA   = dDisLinMVT(dcBA, B,A)
    dB = dB + dBdA
    dA = dA - dBdA
    dBdC   = dDisLinMVT(dcBC, B,C)
    dB = dB + dBdC
    dC = dC - dBdC
    dBdN   = dDisLinMVT(dcBN, B,N)
    dB = dB + dBdN
    dN = dN - dBdN


    # alpha - alpha+
    dApm = dDisLinMVT(dcAAp, A[:,:,:,1:],A[:,:,:,:-1])
    dApm = diffXT(dApm)
    dA = dA + dApm

    # C - alpha+
    dCApm = dDisLinMVT(dcCAp, A[:,:,:, 1:], C[:,:,:,:-1])
    dA[:, :, :, 1:] = dA[:, :, :, 1:] + dCApm
    dC[:, :, :, :-1] = dC[:, :, :, :-1] - dCApm

    # C - N+
    dCNpm = dDisLinMVT(dcCNp,  N[:,:,:, 1:], C[:,:,:, :-1])
    dN[:, :, :, 1:] = dN[:, :, :, 1:] + dCNpm
    dC[:, :, :, :-1] = dC[:, :, :, :-1] - dCNpm

    # alpha - N+
    dCNpm = dDisLinMVT(dcANp,  N[:,:,:, 1:], A[:,:,:, :-1])
    dN[:, :, :, 1:] = dN[:, :, :, 1:] + dCNpm
    dA[:, :, :, :-1] = dA[:, :, :, :-1] - dCNpm


    dX = torch.cat((dA,dC,dN,dB),dim=2)

    return dX


def proj(x3, iter=1, tol=0.5):

    for j in range(iter):

         rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp  = constraints(x3)
         r = torch.cat((rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp), dim=3)
         normr = F.mse_loss(r, torch.zeros_like(r))
         if r.abs().mean() < tol:
             return x3, r
         #print('========== ', j,'    ',  normr.item())

         #lam = dConstraintsTMV(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3)
         lam =  cglsStep(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3, k=50, tol=0.01)

         with torch.no_grad():
            #if j==0:
            alpha = 1.0 #/lam.norm()
            lsiter = 0
            while True:
                xtry = x3 - alpha * lam
                rACt, rNCt, rNAt, rBAt, rBCt, rBNt, rAApt, rCApt, rANpt, rCNpt = constraints(xtry)
                rtry = torch.cat((rACt, rNCt, rNAt, rBAt, rBCt, rBNt, rAApt, rCApt, rANpt, rCNpt), dim=3)

                normrt = F.mse_loss(rtry, torch.zeros_like(r)) #rtry.norm()
                #print(j, lsiter, normrt.item()/normr.item())

                if normrt < normr:
                    break
                alpha = alpha/2
                lsiter = lsiter+1
                if lsiter > 10:
                    break

                if lsiter==0:
                    alpha = alpha*1.5



         x3 = x3 - alpha*lam

    return x3, r


def projCol(x3, iter=1, tol=0.5, contact=4, mu=1e-4 ):
    for j in range(iter):

        d, Q, I, J = collisionDetection(x3, contact)
        gCol = -dcolDetMVT(torch.ones_like(d), x3, Q, I, J)
        x3   = x3 - mu*gCol.t().unsqueeze(0).unsqueeze(0)
        rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp = constraints(x3)
        r = torch.cat((rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp), dim=3)
        normr = F.mse_loss(r, torch.zeros_like(r))
        if r.abs().mean() < tol:
            return x3, r
        # print('========== ', j,'    ',  normr.item())

        # lam = dConstraintsTMV(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3)
        lam = cglsStep(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3, k=50, tol=0.01)

        with torch.no_grad():
            # if j==0:
            alpha = 1.0  # /lam.norm()
            lsiter = 0
            while True:
                xtry = x3 - alpha * lam
                d, Q, I, J = collisionDetection(xtry, contact)
                gCol = -dcolDetMVT(torch.ones_like(d), xtry, Q, I, J)
                xtry = xtry - mu * gCol.t().unsqueeze(0).unsqueeze(0)

                rACt, rNCt, rNAt, rBAt, rBCt, rBNt, rAApt, rCApt, rANpt, rCNpt = constraints(xtry)
                rtry = torch.cat((rACt, rNCt, rNAt, rBAt, rBCt, rBNt, rAApt, rCApt, rANpt, rCNpt), dim=3)

                normrt = F.mse_loss(rtry, torch.zeros_like(r))  # rtry.norm()
                print(j, lsiter, normrt.item()/normr.item(), d.mean().item())

                if normrt < normr:
                    break
                alpha = alpha / 2
                lsiter = lsiter + 1
                if lsiter > 10:
                    break

                if lsiter == 0:
                    alpha = alpha * 1.5

        x3 = xtry

    return x3, r


def cglsStep(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3, k=10, tol=0.01):
    # Solve G'*G*lam = G'*r
    # MVT: dX = dConstraintsTMV(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3)
    # MV: dcAC, dcNC, dcNA, dcBA, dcBC, dcBN, dcAAp, dcCAp, dcANp, dcCNp = dConstraintsMV(dX, x3)

    r = torch.cat((rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp), dim=3)
    normr0 = r.norm()

    s = dConstraintsTMV(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3)
    p = s
    lam = torch.zeros_like(s)
    norms0 = torch.norm(s)
    gamma = norms0**2
    for i in range(k):

        qAC, qNC, qNA, qBA, qBC, qBN, qAAp, qCAp, qANp, qCNp = dConstraintsMV(p, x3)
        q = torch.cat((qAC, qNC, qNA, qBA, qBC, qBN, qAAp, qCAp, qANp, qCNp), dim=3)

        delta = torch.norm(q)**2
        alpha = gamma/delta

        lam = lam + alpha * p
        rAC = rAC - alpha*qAC
        rNC = rNC - alpha*qNC
        rNA = rNA - alpha*qNA
        rBA = rBA - alpha*qBA
        rBC = rBC - alpha*qBC
        rBN = rBN - alpha*qBN
        rAAp = rAAp - alpha*qAAp
        rCAp = rCAp - alpha*qCAp
        rANp = rANp - alpha*qANp
        rCNp = rCNp - alpha*qCNp
        r = torch.cat((rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp), dim=3)
        normr = r.norm()
        #print(i, normr/normr0)
        if normr/normr0 < tol:
            return lam
        s = dConstraintsTMV(rAC, rNC, rNA, rBA, rBC, rBN, rAAp, rCAp, rANp, rCNp, x3)
        norms = torch.norm(s)
        gamma1 = gamma
        gamma = norms**2
        beta = gamma/gamma1
        p = s + beta*p

    return lam


def collisionDetection(X, contact):

    X = X.squeeze().t()
    D =  torch.relu(torch.sum(X**2,dim=1,keepdim=True) + torch.sum(X**2,dim=1,keepdim=True).t() - 2*(X@X.t()))
    U = torch.triu(D,1)
    I, J = torch.where((U <contact) & (U>0))
    Q = U[I,J]
    d = torch.log(Q/contact)
    return d, Q, I, J


def dcolDetMVT(dV, X, Q, I, J):

    X = X.squeeze().t()
    n = X.shape[0]

    dV = dV/Q
    V  = torch.zeros(n,n)
    V[I,J] = dV
    n1 = X.shape[0]
    e2 = torch.ones(12,1)
    e1 = torch.ones(n1,1)
    E12 = e1@e2.t()

    dX = 2*X*(V@E12)  + 2*(X*(V.t()@E12)) - 2*V.t()@X - 2*V@X

    return dX

def JdX(X,dX, Q, contact=0):
    # d(X) = log(f(X))
    # V = df/dX*1/f
    X = X.squeeze().t()

    V = 2*torch.sum(X*dX, dim=1, keepdim=True) + 2*torch.sum(X*dX, dim=1, keepdim=True).t() - \
        2 * (dX @ X.t()) - 2*(X @ dX.t())
    V = torch.triu(V,1)
    I = (V<contact) & (V>0)
    return 1/Q * V[I]

