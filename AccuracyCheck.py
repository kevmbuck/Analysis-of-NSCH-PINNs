# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:37:00 2023

@author: kevbuck
"""

import torch
import numpy as np
from torch.autograd import Variable
from TwoDNSCH import TwoDNSCHPINN

device = 'cpu'

net = TwoDNSCHPINN().to(device)

x1_l = net.x1_l
x1_u = net.x1_u
x2_l = net.x2_l
x2_u = net.x2_u

t0=0
tf = 2

num_tests= 50000

mse_function = torch.nn.MSELoss()

error_vec = []

for i in range(4):
    net.load_state_dict(torch.load(f"model_NSCH_{i}.pt", map_location=torch.device(device)))
    
    x1_collocation = np.random.uniform(low=x1_l, high=x1_u, size=(num_tests,1))
    x2_collocation = np.random.uniform(low=x2_l, high=x2_u, size=(num_tests,1))
    t_collocation = np.random.uniform(low=t0, high=tf, size=(num_tests,1))

    pt_x1_collocation = Variable(torch.from_numpy(x1_collocation).float(), requires_grad=False).to(device)
    pt_x2_collocation = Variable(torch.from_numpy(x2_collocation).float(), requires_grad=False).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=False).to(device)
    
    #compute approximations
    u1_est, u2_est, p_est, phi_est, mu_est = net(pt_x1_collocation, pt_x2_collocation, pt_t_collocation) 
    
    #compute exact solutions
    u1_exact = -pt_x2_collocation * pt_t_collocation
    u2_exact = pt_x1_collocation * pt_t_collocation
    
    phi_exact = torch.sin(pt_x1_collocation*pt_t_collocation) + torch.sin(pt_x2_collocation*pt_t_collocation)
    mu_exact = pt_t_collocation**2 * phi_exact + (phi_exact**2-1) * phi_exact
    
    #compute error
    u1_error_pt = mse_function(u1_est, u1_exact)
    u2_error_pt = mse_function(u2_est, u2_exact)
    
    phi_error_pt = mse_function(phi_est, phi_exact)
    mu_error_pt = mse_function(mu_est, mu_exact)
    
    print(u1_error_pt)
    print(u2_error_pt)
    print(phi_error_pt)
    print(mu_error_pt)
    
    total_error = u1_error_pt + u2_error_pt + phi_error_pt + mu_error_pt
    
    error_vec = np.append(error_vec, total_error.cpu().detach().numpy())
    
np.savetxt('error.txt', error_vec)