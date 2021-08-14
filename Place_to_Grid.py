#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 01:17:55 2021
@author: Yiyuan Zhang
"""

from tqdm import tqdm
import numpy as np
from scipy.linalg import circulant
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle



"""Theoretical representation of grid cell"""
###############################################################################
###############################################################################
def plot_3D_constrain(x, y, Z):
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    xx = np.arange(0,x,1)
    yy = -np.arange(-y,0,1)
    X,Y = np.meshgrid(xx, yy)
    surf = ax.plot_surface(X,Y,Z, cmap='jet')
    fig.colorbar(surf)
    
def gaussian(x, y, c, sigma, mag):
    neigx = np.arange(x)
    neigy = np.arange(y) 
    xx, yy = np.meshgrid(neigx, neigy)
    xx = xx.astype(float)
    yy = yy.astype(float)
    d = 2*sigma*sigma
    ax = np.exp(-np.power(xx-xx.T[c], 2)/d)
    ay = np.exp(-np.power(yy-yy.T[c], 2)/d)
    return mag * (ax * ay).T


def Hippacampas_RF(f, c, gamma1, gamma2, phase_0):
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            a = j-c[0]
            b = -i+c[1]
            if a>0 and b>=0:
                theta = np.arctan(b/a)
            if a<0 and b>0:
                theta = np.arctan(b/a) + np.pi
            if a<0 and b<=0:
                theta = np.arctan(b/a) + np.pi
            if a>0 and b<0:
                theta = np.arctan(b/a) + 2*np.pi
            if a==0 and b>0:
                theta = np.pi/2
            if a==0 and b<0:
                theta = 2*np.pi - np.pi/2
            Z[i,j] =  np.cos(f*theta + phase_0)
    c = (c[1], c[0])
    temp = (gaussian(100,100,c,gamma1,1)-gaussian(100,100,c,gamma2,0.9))
    return Z * temp

f = 3   # 3Hz
P = np.zeros((10000, 1000))
for i in tqdm(range(1000)):
    x = np.random.choice(np.arange(0,100,1))
    y = np.random.choice(np.arange(0,100,1))
    c = (x,y)
    P[:,i] = Hippacampas_RF(f, c, 5, 8, 0).reshape(10000)

Sigma = np.dot(P, P.T)
eigvalue, eigvector = np.linalg.eig(Sigma)

# plot the max eigenvalue's eigenvector
plt.figure(dpi=300)
plt.imshow(eigvector[:,0].reshape(100,100).real)
plt.axis('off')


# Fourier analysis
fft2 = np.fft.fft2(Sigma[4500].reshape(100,100))
shift2center = np.fft.fftshift(fft2)
plot_3D_constrain(100,100, shift2center.real)

plt.figure(figsize=(20,5), dpi=300)
plt.subplot(131)
plt.title('Energy')
plt.imshow(np.abs(shift2center))
plt.colorbar()
plt.axis('off')
plt.subplot(132)
plt.title('Phase')
a = shift2center.real
b = shift2center.imag
plt.imshow(np.arctan2(b,a))
plt.colorbar()
plt.axis('off')
plt.subplot(133)
plt.title('Phase in high energy')
pos=np.where(np.abs(shift2center)>80)
temp = np.zeros((100,100))
temp[pos] = 1
plt.imshow(np.arctan2(b,a)*temp)
plt.colorbar()
plt.axis('off')





"""Neural network model"""
###############################################################################
###############################################################################
import torch
from torch import nn
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\12499\\Desktop\\Grid cell\\')
from gridtorch import scores

def gaussian(x, y, c, sigma, mag):
    neigx = np.arange(x)
    neigy = np.arange(y) 
    xx, yy = np.meshgrid(neigx, neigy)
    xx = xx.astype(float)
    yy = yy.astype(float)
    d = 2*sigma*sigma
    ax = np.exp(-np.power(xx-xx.T[c], 2)/d)
    ay = np.exp(-np.power(yy-yy.T[c], 2)/d)
    return mag * (ax * ay).T

def Hippacampas_RF(c, gamma1, gamma2, phase_0):
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            a = j-c[0]
            b = -i+c[1]
            if a>0 and b>=0:
                theta = np.arctan(b/a)
            if a<0 and b>0:
                theta = np.arctan(b/a) + np.pi
            if a<0 and b<=0:
                theta = np.arctan(b/a) + np.pi
            if a>0 and b<0:
                theta = np.arctan(b/a) + 2*np.pi
            if a==0 and b>0:
                theta = np.pi/2
            if a==0 and b<0:
                theta = 2*np.pi - np.pi/2
            Z[i,j] =  np.cos(3*theta + phase_0)
    c = (c[1], c[0])
    temp = (gaussian(100,100,c,gamma1,1)-gaussian(100,100,c,gamma2,0.9))
    return Z * temp

def grid_scores(grid_cell_rf, mask, center):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    scorer = scores.GridScorer(20, ((-1.1, 1.1), (-1.1, 1.1)), masks_parameters)
    rotated_sacs = scorer.rotated_sacs(grid_cell_rf, scorer._corr_angles, center)
    return scorer.get_grid_scores_for_mask(grid_cell_rf, rotated_sacs, mask)
    
    

P = np.zeros((10000, 1000))
for i in tqdm(range(1000)):
    x = np.random.choice(np.arange(0,100,1))
    y = np.random.choice(np.arange(0,100,1))
    c = (x,y)
    P[:,i] = Hippacampas_RF(c, 10, 20, 0).reshape(10000)
    
    

"RNN model"
###############################################################################
# Hyper Parameters and Data
INPUT_SIZE = 10000      # rnn input size
LR = 0.001           # learning rate
y_np = P
x_np = np.eye(10000)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=100,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            #nonlinearity='relu'
        )
        self.out = nn.Linear(100, 1000)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn = RNN().cuda()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) 
loss_func = nn.MSELoss()
h_state = None

Loss = []
for step in tqdm(range(500)):
    x = torch.from_numpy(x_np[:, np.newaxis]).to(torch.float32).cuda()  
    y = torch.from_numpy(y_np[:, np.newaxis]).to(torch.float32).cuda()

    prediction, h_state = rnn(x, h_state)   # rnn output
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # calculate loss
    Loss.append(loss)
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients
    
plt.figure(dpi=300)
plt.plot(Loss)


# Search grid cell
x = torch.from_numpy(x_np[:, np.newaxis]).to(torch.float32).cuda() 
prediction, h_state = rnn(x, h_state)   # rnn output
for i in range(100):
    plt.figure(dpi=300)
    grid_cell_rf = h_state[0].data[:,i].reshape(100,100).cpu()
    plt.imshow(grid_cell_rf)
    #sco = grid_scores(grid_cell_rf.data.numpy(), np.ones((100,100)), (50,50))[0]
    plt.title(i)
    plt.axis('off')
    



"FC model"
###############################################################################
class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.linear = nn.Sequential(nn.Linear(10000, 100),
                                    nn.Linear(100, 1000))
    def forward(self, x):
        x = self.linear(x)
        return x

model = LinearNN().cuda()
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

y_np = P
x_np = np.eye(10000)

Loss = []
for step in tqdm(range(500)):
    inputs = torch.from_numpy(x_np[:, np.newaxis]).to(torch.float32).cuda()
    target = torch.from_numpy(y_np[:, np.newaxis]).to(torch.float32).cuda()
    out = model(inputs) # 前向传播
    loss = criterion(out, target) # 计算误差
    Loss.append(loss)
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 后向传播
    optimizer.step() # 调整参数
    
plt.figure(dpi=300)
plt.plot(Loss)


# Search grid cell
model = model.cpu()
inputs = torch.from_numpy(x_np[:, np.newaxis]).to(torch.float32)
out = model.linear[0](inputs)
for i in range(100):
    plt.figure(dpi=300)
    plt.imshow(out.data[:,0,i].reshape(100,100))
    plt.title(i)
    plt.axis('off')



