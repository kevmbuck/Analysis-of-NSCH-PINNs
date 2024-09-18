# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:52:04 2023

@author: kevbuck
"""

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

from TwoDNSCH import TwoDNSCHPINN

###########################################

time_plotted = 4

###########################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = TwoDNSCHPINN().to(device)

net.load_state_dict(torch.load("model_NSCHt4.pt", map_location=torch.device('cpu')))

#Graph at various time slices

spatial_discretization = 100

#Define numpy arrays for inputs
x1 = np.linspace(net.x1_l,net.x1_u,spatial_discretization).reshape(spatial_discretization)
x2 = np.linspace(net.x2_l,net.x2_u,spatial_discretization).reshape(spatial_discretization)
x1x2 = np.array(np.meshgrid(x1, x2)).reshape(2,spatial_discretization**2)

t = time_plotted*np.ones((spatial_discretization**2,1))

x1_input = x1x2[0].reshape(spatial_discretization**2, 1)
x2_input = x1x2[1].reshape(spatial_discretization**2, 1)

x1x2 = [x1_input, x2_input]

#convert to pytorch tensors
pt_x1 = Variable(torch.from_numpy(x1_input).float(), requires_grad=False).to(device)
pt_x2 = Variable(torch.from_numpy(x2_input).float(), requires_grad=False).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)

#get network outputs
pt_u1, pt_u2, pt_p, pt_phi, pt_mu = net(pt_x1, pt_x2, pt_t)

#Convert back to numpy
u1, u2, p, phi, mu = pt_u1.data.cpu().numpy(), pt_u2.data.cpu().numpy(), pt_p.data.cpu().numpy(), pt_phi.data.cpu().numpy(), pt_mu.data.cpu().numpy()

X, Y = np.meshgrid(x1, x2)

fig, axs = plt.subplots(2,2)
fig.suptitle(f'Time = {time_plotted}')
fig.tight_layout()
axs[0,0].set_title('u vector field')
axs[0,1].set_title('pressure')
axs[1,0].set_title('phi')
axs[1,1].set_title('mu')
axs[0,0].quiver(x1_input, x2_input, u1, u2)
axs[0,1].pcolor(X, Y, p.reshape(X.shape))
axs[1,0].contour(X, Y, phi.reshape(X.shape), vmin = -1, vmax = 1)
axs[1,1].pcolor(X, Y, mu.reshape(X.shape))

plt.legend()
plt.show()
