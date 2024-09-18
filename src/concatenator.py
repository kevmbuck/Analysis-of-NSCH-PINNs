# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 07:41:51 2023

@author: kevbuck
"""
import numpy as np
import os, sys


error = open('error.txt', 'r')
epsilon = open('epsilon.txt', 'r')
    
epsilon_lines = epsilon.readlines()
error_lines = error.readlines()

epsilon_vec = np.array(epsilon_lines, dtype='double').reshape(2,1)
error_vec = np.array(error_lines, dtype='double').reshape(2,1)

print(epsilon_vec)
print(error_vec)

results = np.concatenate((epsilon_vec, error_vec), axis=1)
print(results)

np.savetxt('results.txt', results)
