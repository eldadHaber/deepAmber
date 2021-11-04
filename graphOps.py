import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##### Scalar graph #################################################
# Operats on a tensor that represents a graph
# V = [Batch, Channel, N]
class scalarGraph(nn.Module):

    def __init__(self, iInd, jInd, nnodes, W=torch.tensor([1.0])):
        super(scalarGraph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes
        device = iInd.device
        W = W.to(device)
        self.W = W

    def nodeGrad(self, x, W=[]):
        if len(W)==0:
            W = self.W
        g = W * (x[:, :, self.iInd] - x[:, :, self.jInd])
        return g

    def nodeAve(self, x, W=[]):
        if len(W)==0:
            W = self.W
        g = W * (x[:, :, self.iInd] + x[:, :, self.jInd]) / 2.0
        return g


    def edgeDiv(self, g, W=[]):
        if len(W)==0:
            W = self.W
        x = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        # z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        # for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        # for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]

        x.index_add_(2, self.iInd, W * g)
        x.index_add_(2, self.jInd, -W * g)

        return x

    def edgeAve(self, g,  W=[], method='ave'):
        if len(W)==0:
            W = self.W
        x1 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        x2 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)

        x1.index_add_(2, self.iInd, W * g)
        x2.index_add_(2, self.jInd, W * g)
        if method == 'max':
            x = torch.max(x1, x2)
        elif method == 'ave':
            x = (x1 + x2) / 2

    def nodeLap(self, x):
        g = self.nodeGrad(x)
        d = self.edgeDiv(g)
        return d

    def nodeProd(self,S):
        # SP = torch.bmm(S[:, :, self.iInd].transpose(2, 0).transpose(2, 1),
        #                S[:, :, self.jInd].transpose(2, 0)).transpose(2, 0)

        SP = S[:, :, self.iInd]*S[:, :, self.jInd]

        return SP

    def edgeLength(self, x):
        g = self.nodeGrad(x)
        #L = torch.sqrt(torch.pow(g, 2).sum(dim=1))
        L = torch.pow(g, 2)

        return L

    def toVector(self):
        G = vectorGraph(self.iInd, self.jInd, self.nnodes)
        return G


########## Vector Graph ##########
# Vectors are defined as V = [Batch, Channel, 9 or 3, N]
# 9 for triangles, 3 for simple points
#

class vectorGraph(nn.Module):

    def __init__(self, iInd, jInd, nnodes, W=torch.tensor([1.0])):
        super(vectorGraph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes
        device = iInd.device
        W = W.to(device)
        self.W = W

    def checkInput(self,x):
        scalar = False
        if len(x.shape)==3:
            x = x.unsqueeze(1)
            scalar = True
        return x, scalar

    def checkOutput(self,x, scalar):
        if scalar==True:
            x = x.squeeze(1)
        return x


    def nodeGrad(self, x, W=[]):
        if len(W)==0:
            W = self.W
        x, scalar = self.checkInput(x)

        g = W * (x[:, :, :, self.iInd] - x[:, :, :, self.jInd])
        g = self.checkOutput(g, scalar)

        return g

    def nodeAve(self, x, W=[]):
        if len(W)==0:
            W = self.W
        x, scalar = self.checkInput(x)

        g = W * (x[:, :, :, self.iInd] + x[:, :, :, self.jInd]) / 2.0
        g = self.checkOutput(g, scalar)

        return g


    def edgeDiv(self, g, W=[]):
        if len(W)==0:
            W = self.W

        g, scalar = self.checkInput(g)
        x = torch.zeros(g.shape[0], g.shape[1], g.shape[2], self.nnodes, device=g.device)
        x = x.to(g.dtype)
        W = W.to(g.dtype)

        x.index_add_(3, self.iInd, W * g)
        x.index_add_(3, self.jInd, -W * g)
        x = self.checkOutput(x, scalar)

        return x

    def edgeAve(self, g,  W=[], method='ave'):
        if len(W)==0:
            W = self.W
        g, scalar = self.checkInput(g)

        x1 = torch.zeros(g.shape[0], g.shape[1], g.shape[2], self.nnodes, device=g.device)
        x2 = torch.zeros(g.shape[0], g.shape[1], g.shape[2], self.nnodes, device=g.device)

        x1.index_add_(3, self.iInd, W * g)
        x2.index_add_(3, self.jInd, W * g)
        if method == 'max':
            x = torch.max(x1, x2)
        elif method == 'ave':
            x = (x1 + x2) / 2

        x = self.checkOutput(x, scalar)
        return x

    def nodeLap(self, x):
        g = self.nodeGrad(x)
        d = self.edgeDiv(g)
        return d

    def edgeLength(self, x):
        g = self.nodeGrad(x)
        L = 1/3*torch.pow(g, 2).sum(dim=2)

        return L

    def getTorsionAngles(self, x):
        # The coords are organized as (Ca, C, N)
        Ca  = x[:, :, :3, self.iInd]
        C   = x[:, :, 3:6, self.iInd]
        N   = x[:, :, 6:9, self.iInd]
        Ca2 = x[:, :, :3, self.jInd]
        C2  = x[:, :, 3:6, self.jInd]
        N2 = x[:, :,  6:9, self.jInd]
        # Compute w = tor(Ca,C,N,Ca)
        omega, cosOmega, sinOmega = torsionAngle(Ca,C,N2,Ca2)

        # Compute phi = tor(C,N,Ca,C2)
        phi, cosPhi, sinPhi = torsionAngle(C,N2,Ca2,C2)

        # Compute psi = tor(N, Cα, C, N2)
        psi, cosPsi, sinPsi = torsionAngle(N, Ca, C, N2)

        tor = torch.cat((omega, phi, psi), dim=2)
        torS = torch.cat((cosOmega, sinOmega, cosPhi, sinPhi,  cosPsi, sinPsi), dim=2)
        return tor, torS


    def nodeProd(self,S):
        # S = assumed to be a scalar
        #SP = torch.bmm(S[:, :, self.iInd].transpose(2, 0).transpose(2, 1),
        #               S[:, :, self.jInd].transpose(2, 0)).transpose(2, 0)
        SP = S[:, :, self.iInd]*S[:, :, self.jInd]
        return SP


### Graph Functions ############################################



def vectorDotProd(V, W):
    # V = [B, C, 9/3  N]
    # W = [B, C, 9/3, N]
    Vout = torch.sum(V*W, dim=2)

    return Vout

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

