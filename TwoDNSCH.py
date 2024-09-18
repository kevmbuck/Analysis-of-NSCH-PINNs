# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:05:35 2023

@author: kevbuck
"""

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class TwoDNSCHPINN(nn.Module):
    #1 layer N node Neural Network
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.mse_cost_function = torch.nn.MSELoss()
        
        N = 200
        
        #input x1, x2, t
        self.layer_u1_1 = nn.Linear(3, N)
        self.layer_u1_2 = nn.Linear(N,N)
        self.layer_u1_3 = nn.Linear(N,N)
        self.layer_u1_4 = nn.Linear(N,N)
        self.layer_u1_5 = nn.Linear(N,N)
        self.layer_u1_out = nn.Linear(N,1)
        
        self.layer_u2_1 = nn.Linear(3, N)
        self.layer_u2_2 = nn.Linear(N,N)
        self.layer_u2_3 = nn.Linear(N,N)
        self.layer_u2_4 = nn.Linear(N,N)
        self.layer_u2_5 = nn.Linear(N,N)
        self.layer_u2_out = nn.Linear(N,1)
        
        self.layer_p_1 = nn.Linear(3,N)
        self.layer_p_2 = nn.Linear(N,N)
        self.layer_p_3 = nn.Linear(N,N)
        self.layer_p_4 = nn.Linear(N,N)
        self.layer_p_5 = nn.Linear(N,N)
        self.layer_p_out = nn.Linear(N,1)
        
        self.layer_phi_1 = nn.Linear(3,N)
        self.layer_phi_2 = nn.Linear(N,N)
        self.layer_phi_3 = nn.Linear(N,N)
        self.layer_phi_4 = nn.Linear(N,N)
        self.layer_phi_5 = nn.Linear(N,N)
        self.layer_phi_out = nn.Linear(N,1)
        
        self.layer_mu_1 = nn.Linear(3,N)
        self.layer_mu_2 = nn.Linear(N,N)
        self.layer_mu_3 = nn.Linear(N,N)
        self.layer_mu_4 = nn.Linear(N,N)
        self.layer_mu_5 = nn.Linear(N,N)
        self.layer_mu_out = nn.Linear(N,1)
        #output u1, u2, p

        self.inner_optimizer = torch.optim.Adam(self.parameters())
        
        self.x1_l = 0
        self.x1_u = np.pi
        self.x2_l = 0
        self.x2_u = np.pi
    
    def forward(self, x1, x2, t):
        x = torch.cat([x1, x2, t],axis=1) # combined three arrays of 1 columns each to one array of 3 columns
        x = self.flatten(x)
        
        u1 = torch.sigmoid(self.layer_u1_1(x))
        u1 = torch.sigmoid(self.layer_u1_2(u1))
        u1 = torch.sigmoid(self.layer_u1_3(u1))
        u1 = torch.sigmoid(self.layer_u1_4(u1))
        u1 = torch.sigmoid(self.layer_u1_5(u1))
        u1 = self.layer_u2_out(u1)
        
        u2 = torch.sigmoid(self.layer_u2_1(x))
        u2 = torch.sigmoid(self.layer_u2_2(u2))
        u2 = torch.sigmoid(self.layer_u2_3(u2))
        u2 = torch.sigmoid(self.layer_u2_4(u2))
        u2 = torch.sigmoid(self.layer_u2_5(u2))
        u2 = self.layer_u2_out(u2)
        
        p = torch.sigmoid(self.layer_p_1(x))
        p = torch.sigmoid(self.layer_p_2(p))
        p = torch.sigmoid(self.layer_p_3(p))
        p = torch.sigmoid(self.layer_p_4(p))
        p = torch.sigmoid(self.layer_p_5(p))
        p = self.layer_p_out(p)
        
        phi = torch.sigmoid(self.layer_phi_1(x))
        phi = torch.sigmoid(self.layer_phi_2(phi))
        phi = torch.sigmoid(self.layer_phi_3(phi))
        phi = torch.sigmoid(self.layer_phi_4(phi))
        phi = torch.sigmoid(self.layer_phi_5(phi))
        phi = self.layer_phi_out(phi)
        
        mu = torch.sigmoid(self.layer_mu_1(x))
        mu = torch.sigmoid(self.layer_mu_2(mu))
        mu = torch.sigmoid(self.layer_mu_3(mu))
        mu = torch.sigmoid(self.layer_mu_4(mu))
        mu = torch.sigmoid(self.layer_mu_5(mu))
        mu = self.layer_mu_out(mu)
               
        return u1, u2, p, phi, mu
    
    def nu(self, phi):
        nu_1 = 1
        nu_2 = 1
        nu = nu_1 * (1-phi) + nu_2 * (1 + phi)
        nu = .5 * nu
        return nu
    
    def nu_prime(self, phi):
        nu_1 = 1
        nu_2 = 1
        nu_prime = (nu_2-nu_1)/2
        return nu_prime
    
    def Psi_prime(self, phi):
        Psi_prime = (phi**2 - 1) * phi
        return Psi_prime
    
    def Psi_double_prime(self, phi):
        psi_double_prime = 3*phi**2 - 1
        return psi_double_prime
    
    def Psi_triple_prime(self, phi):
        triple = 6 * phi
        return triple
    
    def ns_x1_forcing(self, x1, x2, t):
        phi = torch.sin(x1*t) + torch.sin(x2*t)
        mu = (t)**2 * phi + self.Psi_prime(phi)
        force = -x2 - x1 * t**2 - mu * t * torch.cos(x1*t)
        
        return force
    
    def ns_x2_forcing(self, x1, x2, t):
        phi = torch.sin(x1*t) + torch.sin(x2*t)
        mu = (t)**2 * phi + self.Psi_prime(phi)
        force = x1 - x2 * t**2 - mu * t * torch.cos(x2*t)

        return force
    
    def ch_forcing(self, x1, x2, t):
        phi = torch.sin(x1*t) + torch.sin(x2*t)
        phi_t = x1 * torch.cos(x1 * t) + x2 * torch.cos(x2 * t)
        phi_x1 = t * torch.cos(x1 * t)
        phi_x1x1 = -t**2 * torch.sin(t * x1)
        phi_x2 = t * torch.cos(x2 * t)
        phi_x2x2 = -t**2 * torch.sin(t * x2)
        mu = t**2*phi + self.Psi_prime(phi)
        mu_x1 = t**2 * phi_x1 + self.Psi_double_prime(phi) * phi_x1
        mu_x1x1 = t**2 * phi_x1x1 + self.Psi_triple_prime(phi) * phi_x1**2 + self.Psi_double_prime(phi) * phi_x1x1
        mu_x2 = t**2 * phi_x2 + self.Psi_double_prime(phi) * phi_x2
        mu_x2x2 = t**2 * phi_x2x2 + self.Psi_triple_prime(phi) * phi_x2**2 + self.Psi_double_prime(phi) * phi_x2x2
        
        force = phi_t - t * x2 * phi_x1 + t * x1 * phi_x2 - self.nu_prime(phi) * phi_x1 * mu_x1 - self.nu_prime(phi) * phi_x2 * mu_x2 - self.nu(phi) * (mu_x1x1 + mu_x2x2)
        return force
    
    def PDE_Loss(self, x1, x2, t):
        u1, u2, p, phi, mu = self(x1, x2, t)
               
        u1_x1 = torch.autograd.grad(u1.sum(), x1, create_graph=True)[0]
        u1_x1x1 = torch.autograd.grad(u1_x1.sum(), x1, create_graph=True)[0]
        
        u1_x2 = torch.autograd.grad(u1.sum(), x2, create_graph=True)[0]
        u1_x2x2 = torch.autograd.grad(u1_x2.sum(), x2, create_graph=True)[0]
        
        u2_x1 = torch.autograd.grad(u2.sum(), x1, create_graph=True)[0]
        u2_x1x1 = torch.autograd.grad(u2_x1.sum(), x1, create_graph=True)[0]
        
        u2_x2 = torch.autograd.grad(u2.sum(), x2, create_graph=True)[0]
        u2_x2x2 = torch.autograd.grad(u1_x2.sum(), x2, create_graph=True)[0]
        
        u2_x2x1 = torch.autograd.grad(u2_x1.sum(), x2, create_graph=True)[0]
        u1_x1x2 = torch.autograd.grad(u1_x2.sum(), x1, create_graph=True)[0]
        
        u1_t = torch.autograd.grad(u1.sum(), t, create_graph=True)[0]
        u2_t = torch.autograd.grad(u2.sum(), t, create_graph=True)[0]

        p_x1 = torch.autograd.grad(p.sum(), x1, create_graph=True)[0]
        p_x2 = torch.autograd.grad(p.sum(), x2, create_graph=True)[0]
        
        phi_x1 = torch.autograd.grad(phi.sum(), x1, create_graph=True)[0]  
        phi_x1x1 = torch.autograd.grad(phi_x1.sum(), x1, create_graph=True)[0]
        phi_x2 = torch.autograd.grad(phi.sum(), x2, create_graph=True)[0]
        phi_x2x2 = torch.autograd.grad(phi_x2.sum(), x2, create_graph=True)[0]
        phi_t = torch.autograd.grad(phi.sum(), t, create_graph=True)[0]
        
        mu_x1 = torch.autograd.grad(mu.sum(), x1, create_graph=True)[0]
        mu_x1x1 = torch.autograd.grad(mu_x1.sum(), x1, create_graph=True)[0]
        mu_x2 = torch.autograd.grad(mu.sum(), x2, create_graph=True)[0]
        mu_x2x2 = torch.autograd.grad(mu_x2.sum(), x2, create_graph=True)[0]
                
        #compute loss
        div1 = self.nu(phi) * u1_x1x1 + self.nu_prime(phi) * phi_x1 * u1_x1 + .5 * self.nu(phi) * (u1_x2x2 + u2_x2x1) + (u1_x2 + u2_x1) * self.nu_prime(phi) * phi_x2 * .5
        div2 = self.nu(phi) * u2_x2x2 + self.nu_prime(phi) * phi_x2 * u2_x2 + .5 * self.nu(phi) * (u2_x1x1 + u1_x1x2) + (u1_x2 + u2_x1) * self.nu_prime(phi) * phi_x1 * .5
        
        ns_x1_loss = u1_t + u1*u1_x1 + u2*u1_x2 - div1 + p_x1 - mu*phi_x1 - self.ns_x1_forcing(x1, x2, t)
        ns_x2_loss = u2_t + u1*u2_x1 + u2*u2_x2 - div2 + p_x2 - mu*phi_x2 - self.ns_x2_forcing(x1, x2, t)
        
        ch1_rhs = self.nu_prime(phi) * (phi_x1 * mu_x1 + phi_x2 * mu_x2) + self.nu(phi) * (mu_x1x1 + mu_x2x2)
        
        ch_loss_1 = phi_t + u1 * phi_x1 + u2 * phi_x2 - ch1_rhs - self.ch_forcing(x1, x2, t)
        ch_loss_2 = mu + phi_x1x1 + phi_x2x2 - self.Psi_prime(phi)
        
        zeros = torch.zeros_like(ns_x1_loss)
        
        return self.mse_cost_function(ns_x1_loss, zeros) + self.mse_cost_function(ns_x2_loss, zeros) + self.mse_cost_function(ch_loss_1, zeros) + self.mse_cost_function(ch_loss_2, zeros)
    
    def Initial_Condition_u1(self, x1, x2):
        #return torch.cos(x1*np.pi*2)*torch.cos(x2*np.pi*3)
        #nu =1/40
        #lam = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2) 
        #u1 = torch.ones_like(x1) - torch.exp(lam * x1)*torch.cos(2*np.pi*x2)
        return torch.zeros_like(x1)
    
    def Initial_Condition_u2(self, x1, x2):
        #return torch.cos(x1*np.pi*2)*torch.cos(x2*np.pi*3)
        #nu =1/40
        #lam = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2) 
        #u2 = lam/(2*np.pi) * torch.exp(lam*x1) * torch.sin(2*np.pi*x2)
        return torch.zeros_like(x1)
    
    def Initial_Condition_phi(self, x1, x2):
        phi_0 = torch.zeros_like(x1)
        return phi_0
    
    def Initial_Condition_mu(self, x1, x2):
        #only works if laplacian of phi_0 = 0, otherwise need to put in manually
        phi_0 = self.Initial_Condition_phi(x1, x2)
        mu_0 = self.Psi_prime(phi_0)
        return mu_0
    
    def Initial_Condition_Loss(self, x1, x2):
        t = Variable(torch.zeros_like(x1), requires_grad=False).to(device)
        
        u1_pred, u2_pred, _, phi_pred, mu_pred = self(x1, x2, t)
        
        u1_exact = self.Initial_Condition_u1(x1, x2)
        u2_exact = self.Initial_Condition_u2(x1, x2)
        phi_exact = self.Initial_Condition_phi(x1, x2)
        mu_exact = self.Initial_Condition_mu(x1, x2)
        
        initial_condition_loss = self.mse_cost_function(u1_pred, u1_exact) + self.mse_cost_function(u2_pred, u2_exact) + self.mse_cost_function(phi_pred, phi_exact) + self.mse_cost_function(mu_pred, mu_exact)
        
        return initial_condition_loss
    
    def Boundary_Loss(self, x1, x2, t):
        
        x1_l_pt = Variable(self.x1_l * torch.ones_like(t), requires_grad=True).to(device)
        x1_u_pt = Variable(self.x1_u * torch.ones_like(t), requires_grad=True).to(device)
        x2_l_pt = Variable(self.x2_l * torch.ones_like(t), requires_grad=True).to(device)
        x2_u_pt = Variable(self.x2_u * torch.ones_like(t), requires_grad=True).to(device)
        
        #Evaluate at the 4 boundaries
        u1_lower, u2_lower, _, phi_lower, mu_lower = self(x1, x2_l_pt, t)
        u1_upper, u2_upper, _, phi_upper, mu_upper = self(x1, x2_u_pt, t)
        u1_left, u2_left, _, phi_left, mu_left = self(x1_l_pt, x2, t)
        u1_right, u2_right, _, phi_right, mu_right = self(x1_u_pt, x2, t)
        
        #homogeneous neumann on phi and mu
        phi_lower_x2 = torch.autograd.grad(phi_lower.sum(), x2_l_pt, create_graph=True)[0]
        phi_upper_x2 = torch.autograd.grad(phi_upper.sum(), x2_u_pt, create_graph=True)[0]
        phi_left_x1 = torch.autograd.grad(phi_left.sum(), x1_l_pt, create_graph=True)[0]
        phi_right_x1 = torch.autograd.grad(phi_right.sum(), x1_u_pt, create_graph=True)[0]
        
        mu_lower_x2 = torch.autograd.grad(mu_lower.sum(), x2_l_pt, create_graph=True)[0]
        mu_upper_x2 = torch.autograd.grad(mu_upper.sum(), x2_u_pt, create_graph=True)[0]
        mu_left_x1 = torch.autograd.grad(mu_left.sum(), x1_l_pt, create_graph=True)[0]
        mu_right_x1 = torch.autograd.grad(mu_right.sum(), x1_u_pt, create_graph=True)[0]
        
        #for test problem, evaluate functions at the boundary.
        u1_lower_exact = -x2_l_pt * t
        u2_lower_exact = x1 * t
        phi_lower_exact_x2 = t * torch.cos(x2_l_pt * t) #note we do not need a negative, since we examine the x2 derivative instead of the exact normal
        phi_lower_exact = torch.sin(x1 * t) + torch.sin(x2_l_pt * t)
        mu_lower_exact_x2 = t**2* phi_lower_exact_x2 + self.Psi_double_prime(phi_lower_exact)* phi_lower_exact_x2
        
        u1_upper_exact = -x2_u_pt * t
        u2_upper_exact = x1 * t
        phi_upper_exact_x2 = t * torch.cos(x2_u_pt * t) 
        phi_upper_exact = torch.sin(x1 * t) + torch.sin(x2_u_pt * t)
        mu_upper_exact_x2 = t**2* phi_upper_exact_x2 + self.Psi_double_prime(phi_upper_exact)* phi_upper_exact_x2 #
        
        u1_left_exact = -x2 * t
        u2_left_exact = x1_l_pt * t
        phi_left_exact_x1 = t * torch.cos(x1_l_pt * t)
        phi_left_exact = torch.sin(x1_l_pt * t) + torch.sin(x2 * t)
        mu_left_exact_x1 = t**2 * phi_left_exact_x1 + self.Psi_double_prime(phi_left_exact) * phi_left_exact_x1
        
        u1_right_exact = -x2 * t
        u2_right_exact = x1_u_pt * t
        phi_right_exact_x1 = t * torch.cos(x1_u_pt * t)
        phi_right_exact = torch.sin(x1_u_pt * t) + torch.sin(x2 * t)
        mu_right_exact_x1 = t**2 * phi_right_exact_x1 + self.Psi_double_prime(phi_right_exact) * phi_right_exact_x1
              
        zero = torch.zeros_like(t)
        
        #homogenous dirichlet on u
        ####u1_bc_loss = self.mse_cost_function(u1_lower, zero) + self.mse_cost_function(u1_upper, zero) + self.mse_cost_function(u1_left, zero) + self.mse_cost_function(u1_right, zero)
        ####u2_bc_loss = self.mse_cost_function(u2_lower, zero) + self.mse_cost_function(u2_upper, zero) + self.mse_cost_function(u2_left, zero) + self.mse_cost_function(u2_right, zero)
        
        u1_bc_loss = self.mse_cost_function(u1_lower, u1_lower_exact) + self.mse_cost_function(u1_upper, u1_upper_exact) + self.mse_cost_function(u1_left, u1_left_exact) + self.mse_cost_function(u1_right, u1_right_exact)
        u2_bc_loss = self.mse_cost_function(u2_lower, u2_lower_exact) + self.mse_cost_function(u2_upper, u2_upper_exact) + self.mse_cost_function(u2_left, u2_left_exact) + self.mse_cost_function(u2_right, u2_right_exact)
        
        
        #neumann
        phi_n_loss = self.mse_cost_function(phi_lower_x2, phi_lower_exact_x2) + self.mse_cost_function(phi_upper_x2, phi_upper_exact_x2) + self.mse_cost_function(phi_left_x1, phi_left_exact_x1) + self.mse_cost_function(phi_right_x1, phi_right_exact_x1)
        mu_n_loss = self.mse_cost_function(mu_lower_x2, mu_lower_exact_x2) + self.mse_cost_function(mu_upper_x2, mu_upper_exact_x2) + self.mse_cost_function(mu_left_x1, mu_left_exact_x1) + self.mse_cost_function(mu_right_x1, mu_right_exact_x1)
        
        #total loss
        boundary_loss = u1_bc_loss + u2_bc_loss + phi_n_loss + mu_n_loss
                
        return boundary_loss
    
    def Divergence_Loss(self, x1, x2, t):
        u1, u2, _, _, _ = self(x1, x2, t)
        
        u1_x1 = torch.autograd.grad(u1.sum(), x1, create_graph=True)[0]
        u2_x2 = torch.autograd.grad(u2.sum(), x2, create_graph=True)[0]
        
        div = u1_x1 + u2_x2
        
        zero = torch.zeros_like(div)
        
        divergence_loss = self.mse_cost_function(div, zero)
        
        return divergence_loss
        