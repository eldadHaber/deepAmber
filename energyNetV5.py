import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


#from src \
import graphOps as GO
#from src \
import utils

def intTanh(x):
    # s always has real part >= 0
    s = torch.sign(x) * x
    p = torch.exp(-2 * s)
    return s + torch.log1p(p) - np.log(2)

def tanhsq(x, a=0.1):
    # s always has real part >= 0
    return torch.tanh(a*x)**2

def dtanhsq(x, a=0.1):
    # s always has real part >= 0
    return 2*a*torch.tanh(a*x)*(1-torch.tanh(a*x)**2)


def doubleLayer(x, K1, K2):
    x = F.conv1d(x, K1.unsqueeze(2))
    x = F.instance_norm(x)
    x = torch.tanh(x)

    x = F.conv1d(x, K2.unsqueeze(2))

    return x

def getBondAngle(T, Mask):

    n = T.shape
    T = T.reshape(n[0],n[1],3,4,n[3])
    A = T[:,:,:,0,:]
    dA = A[:, :, :, 1:] - A[:,:,:,:-1]
    Am = dA[:, :, :,:-1]
    Ap = dA[:, :, :, 1:]

    n1 = Am/torch.sqrt(torch.sum(Am**2,dim=2,keepdim=True)+1e-3)
    n2 = Ap/torch.sqrt(torch.sum(Ap**2,dim=2,keepdim=True)+1e-3)
    Cp = torch.zeros_like(A[:,:,0,:])
    Cp[:,:,1:-1] = torch.sum(n1*n2,dim=2)
    Cp = Mask*Cp
    return Cp

def vectorCrossProd(n1, n2):
    # V1 = [B, C, 3  N]
    # V2 = [B, C, 3, N]
    # vy*wz - vz*wy
    # vz*wx - vx*wz
    # vx*wy - vy*wx

    Cx = (n1[:, :, 1, :] * n2[:, :, 2, :] - n1[:, :, 2, :] * n2[:, :, 1, :]).unsqueeze(2)
    Cy = (n1[:, :, 2, :] * n2[:, :, 0, :] - n1[:, :, 0, :] * n2[:, :, 2, :]).unsqueeze(2)
    Cz = (n1[:, :, 0, :] * n2[:, :, 1, :] - n1[:, :, 1, :] * n2[:, :, 0, :]).unsqueeze(2)

    C = torch.cat((Cx, Cy, Cz), dim=2)

    return C


def torsionAngle(V1,V2,V3,V4):

    A = V2 - V1
    B = V3 - V2
    C = V4 - V3

    Bsq = torch.relu(torch.sum(B * B, dim=2, keepdim=True))
    AC  = torch.sum(A * C, dim=2, keepdim=True)
    AB  = torch.sum(A * B, dim=2, keepdim=True)
    BC  = torch.sum(B * C, dim=2, keepdim=True)
    x   = -torch.sum(Bsq*AC, dim=2, keepdim=True) + torch.sum(AB*BC, dim=2, keepdim=True)

    absB = torch.sqrt(Bsq).sum(dim=2, keepdim=True)
    BxC  = vectorCrossProd(B, C)
    y    = torch.sum((absB*A)*BxC, dim=2, keepdim=True)

    cosTheta = x/torch.sqrt(x**2 + y**2 + 1e-3)
    sinTheta = y/torch.sqrt(x**2 + y**2 + 1e-3)
    theta = torch.arccos(cosTheta)
    theta = theta*torch.sign(y)
    return theta, cosTheta, sinTheta

def getTorsionAngles(x):
    # The coords are organized as (Ca, C, N)
    nnodes = x.shape[-1]
    Ca  = x[:, :, :3, :-1]
    C   = x[:, :, 3:6, :-1]
    N   = x[:, :, 6:9, :-1]
    Ca2 = x[:, :, :3, 1:]
    C2  = x[:, :, 3:6, 1:]
    N2  = x[:, :, 6:9, 1:]
    # Compute w = tor(Ca,C,N,Ca)
    omega, cosOmega, sinOmega = torsionAngle(Ca,C,N2,Ca2)

    # Compute phi = tor(C,N,Ca,C2)
    phi, cosPhi, sinPhi = torsionAngle(C,N2,Ca2,C2)

    # Compute psi = tor(N, Cα, C, N2)
    psi, cosPsi, sinPsi = torsionAngle(N, Ca, C, N2)

    #tor = torch.cat((omega, phi, psi), dim=2)

    phi   = phi[0,0,0,:-1].squeeze()
    cosPhi = cosPhi[0,0,0,:-1].squeeze()
    sinPhi = sinPhi[0,0,0,:-1].squeeze()

    psi   = psi[0,0,0, 1:].squeeze()
    cosPsi = cosPsi[0,0,0, 1:].squeeze()
    sinPsi = sinPsi[0,0,0, 1:].squeeze()

    omega = omega[0,0,0, 1:].squeeze()
    cosOmega = cosOmega[0,0,0, 1:].squeeze()
    sinOmega = sinOmega[0,0,0, 1:].squeeze()


    tor = torch.zeros(3,nnodes)
    tor[0, 1:-1] = phi
    tor[1, 1:-1] = psi
    tor[2, 1:-1] = omega

    torS = torch.stack((cosOmega, sinOmega, cosPhi, sinPhi,  cosPsi, sinPsi), dim=0)
    torS = torch.cat((torch.zeros(6,1), torS, torch.zeros(6,1)),dim=1)
    return tor, torS


def getNodeDistance(T, Graph, Mask=[]):
    # Energy of all the nonconstant distances
    # Data organized as A C N B

    if len(Mask) == 0:
        Mask = torch.ones(1,1,T.shape[3])

    MaskE = Graph.nodeAve(Mask)
    inValidInd = MaskE<1

    n = T.shape
    nnodes = n[-1]

    T = T.reshape(n[0], n[1], 4, 3, nnodes)

    I = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=T.device)
    J = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], device=T.device)
    # K = [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    TI = T[:, :, I, :, :]
    TJ = T[:, :, J, :, :]
    G = TI[:, :, :, :, Graph.iInd] - TJ[:, :, :, :, Graph.jInd]

    f = torch.sum(G ** 2, dim=3)

    f[:, :, :, inValidInd.squeeze()] = 0
    return f



class energyGraphNetwork(nn.Module):

    def __init__(self, Nopen, nlayer, h=0.1):
        super(energyGraphNetwork, self).__init__()

        self.h = h
        nodeFeatIn = 40+6  # PSSM+Seq+Tor
        EdgeFeatIn = 16    # distances
        self.K1Nopen = nn.Parameter(torch.randn(Nopen, nodeFeatIn))
        self.K2Nopen = nn.Parameter(torch.randn(Nopen, Nopen))

        self.K1Eopen = nn.Parameter(torch.randn(Nopen, EdgeFeatIn))
        self.K2Eopen = nn.Parameter(torch.randn(Nopen, Nopen))

        nopen      = 3*Nopen
        self.nopen = nopen

        Id  = (torch.cat((torch.eye(nopen,nopen),torch.eye(nopen,nopen)), dim=1)).unsqueeze(0)
        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        self.KE = nn.Parameter(IdTensor)

        #self.KE = nn.Parameter(torch.randn(nlayer, nopen, 2*nopen))

    def forward(self, SeqData, Coords, Graph, M=torch.ones(1)):

        ME = torch.ones(1)
        if len(M)>1:
            ME = Graph.nodeGrad(M)
            ME[ME<1] = 0

        xe = getNodeDistance(Coords, Graph, Mask=[])
        xe = xe - torch.mean(xe, dim=3, keepdim=True)
        _, xn = getTorsionAngles(Coords)
        xn = torch.cat((xn.unsqueeze(0), SeqData),dim=1)
        #xn = SeqData
        xn = doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = doubleLayer(xe.squeeze(1), self.K1Eopen, self.K2Eopen)

        xn = torch.cat([xn, Graph.edgeDiv(xe), Graph.edgeAve(xe)], dim=1)
        xn = xn - xn.mean(dim=2, keepdim=True)

        nlayers = self.KE.shape[0]

        for i in range(nlayers):

            gradX   = ME*Graph.nodeGrad(xn)
            intX    = ME*Graph.nodeAve(xn)

            xe = torch.cat([gradX, intX], dim=1)
            #xe  = doubleLayer(xe, self.KE1[i], self.KE2[i])
            xe = doubleLayer(xe, self.KE[i], self.KE[i].t())
            divE = M*Graph.edgeDiv(xe[:,:self.nopen,:])
            aveE = M*Graph.edgeAve(xe[:,self.nopen:2*self.nopen,:])

            xn   = M*(xn - self.h * (divE + aveE))


        xn = xn - xn.mean(dim=2, keepdim=True)

        return xn

def colisionDetection(P, M, contact=6.0):

    ind = M.squeeze() > 0
    P = P[:, :, ind].squeeze(0)
    p = torch.sum(P ** 2, dim=0, keepdim=True)
    DP = torch.triu(torch.sqrt(torch.relu(p + p.t() - 2 * P.t() @ P)), 1)

    con = (DP < contact)

    DP = DP[con]
    a = (DP > 0)
    DP = DP[a]

    ecol = -torch.log(DP/contact)
    ecol = ecol.sum()

    return ecol


def updateCoords(enet, X0, xnS, Graph, lrX=1e-1, iter=100, Emin=0, ratio=0.8):
    ###### Optimize over X - Generator
    Xp = torch.clone(X0).detach()
    Xp.requires_grad = True
    optimizerE = optim.SGD([{'params': Xp, 'lr': lrX}])
    M = torch.ones(1,1, X0.shape[-1])
    for i in range(iter):
        optimizerE.zero_grad()
        #Xpp, res = cons.proj(Xp, 100, 1e-2)
        Xpp = Xp
        ep = enet(xnS,Xpp, Graph)
        Ep = 0.5*(ep**2).mean()
        Ec = colisionDetection(Xpp.squeeze(1), M, contact=6.0)

        Etotal = Ep + Ec
        Etotal.backward()
        torch.nn.utils.clip_grad_value_(Xp, 0.1)
        optimizerE.step()

        dRMSD = utils.lossFundRMSD(Xpp.squeeze(0), X0.squeeze(0), M, contact=1e3)
        dX    = torch.mean(torch.abs(Xp-X0))
        print('      Eiter =  %2d   rloss = %3.2e    Ep = %3.2e      Ec = %3.2e   dRMSD = %3.2e   |dX| = %3.2e'%
              (i, Emin/Etotal, Ep, Ec, dRMSD, dX))
        if Etotal < ratio*Emin:
            return Xp, Ep
    return Xp, Etotal
