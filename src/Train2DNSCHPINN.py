# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:51:00 2023

@author: kevbuck
"""

import torch
import numpy as np
from torch.autograd import Variable
import time



from TwoDNSCH import TwoDNSCHPINN


def create_network():
    start = time.time()
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = TwoDNSCHPINN().to(device)

    time_slices = np.array([.1,.5,1,1.5,2])
    global epsilon
    epsilon = []
    
    for i in range(4):
        #we will optimize with progressively smaller learning rates
        if i == 0:
            iterations = 10000
            learning_rate = 10**-3
        elif i == 1:
            time_slices = [2]
            iterations = 20000
            learning_rate = 10**-4
        elif i == 2:
            learning_rate = 10**-5
        elif i ==3:
            iterations = 10000
            learning_rate = 10**-6
        training_loop(net, time_slices, iterations, learning_rate)
        torch.save(net.state_dict(), f"model_NSCH_{i}.pt")

    np.savetxt('epsilon.txt', epsilon)
    
    end = time.time()

    print("Time Elapsed:\t", end-start)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def training_loop(net, time_slices, iterations, learning_rate):
    global epsilon
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
      
    #update learning rate
    for g in net.inner_optimizer.param_groups:
        g['lr'] = learning_rate
    
    #get domain boundaries
    x1_l = net.x1_l
    x1_u = net.x1_u
    x2_l = net.x2_l
    x2_u = net.x2_u
    
    #start at time 0, upper time updates in slices
    t_l = 0

    BC_collocation = int(500)
    IC_collocation = int(500)
    pde_collocation_pts = int(1000)
    
    i=0
    for final_time in time_slices: 
        with torch.autograd.no_grad():
            print("Current Final Time:", final_time, "Current Learning Rate: ", get_lr(net.inner_optimizer))  
        
        #Iterate over these points
        for epoch in range(iterations+1):
            
            #Select Collocation Points
            ##Initial Condition Points
            x1_ic = np.random.uniform(low=x1_l, high=x1_u, size=(IC_collocation,1))
            x2_ic = np.random.uniform(low=x2_l, high=x2_u, size=(IC_collocation,1))

            pt_x1_ic = Variable(torch.from_numpy(x1_ic).float(), requires_grad=False).to(device)
            pt_x2_ic = Variable(torch.from_numpy(x2_ic).float(), requires_grad=False).to(device)
        
            ##Boundary Condition Points
            x1_bc = np.random.uniform(low=x1_l, high=x1_u, size=(BC_collocation,1))
            x2_bc = np.random.uniform(low=x2_l, high=x2_u, size=(BC_collocation,1))       
            t_bc = np.random.uniform(low=t_l, high=final_time, size=(BC_collocation,1))

            pt_x1_bc = Variable(torch.from_numpy(x1_bc).float(), requires_grad=True).to(device)
            pt_x2_bc = Variable(torch.from_numpy(x2_bc).float(), requires_grad=True).to(device)
            pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
        
            ##PDE Domain Points
            x1_collocation = np.random.uniform(low=x1_l, high=x1_u, size=(pde_collocation_pts,1))
            x2_collocation = np.random.uniform(low=x2_l, high=x2_u, size=(pde_collocation_pts,1))
            t_collocation = np.random.uniform(low=t_l, high=final_time, size=(pde_collocation_pts,1))

            pt_x1_collocation = Variable(torch.from_numpy(x1_collocation).float(), requires_grad=True).to(device)
            pt_x2_collocation = Variable(torch.from_numpy(x2_collocation).float(), requires_grad=True).to(device)
            pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
            
            
            ###Actual stepping loop
            
            #zero the gradient
            net.inner_optimizer.zero_grad()
            
            #Loss based on Dirichlet Boundary Condition
            mse_bc = net.Boundary_Loss(pt_x1_bc, pt_x2_bc, pt_t_bc)
            
            #Loss based on Initial Condition
            mse_ic = net.Initial_Condition_Loss(pt_x1_ic, pt_x2_ic)
    
            #Loss based on PDE
            mse_f = net.PDE_Loss(pt_x1_collocation, pt_x2_collocation, pt_t_collocation)
    
            #Loss based on divergence (same collocation as PDE)
            mse_div = net.Divergence_Loss(pt_x1_collocation, pt_x2_collocation, pt_t_collocation)
            
            #Combine Loss functions
            loss = 100*mse_ic + mse_f + mse_bc + 10*mse_div
    
            loss.backward()
            net.inner_optimizer.step()
    
            #Print Loss every 500 Epochs
            with torch.autograd.no_grad():
                if epoch%1000 == 0: 
                    print("Iteration:", epoch, "Total Loss:", loss.data)
                    print("\tIC Loss: ", mse_ic.data, "\tPDE Loss: ", mse_f.data, "\tBC Loss: ", mse_bc.data, "\tDiv Loss: ", mse_div.data)
            
            i = i + 1    
    epsilon = np.append(epsilon, loss.cpu().detach().numpy())


create_network()