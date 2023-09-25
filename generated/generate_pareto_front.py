# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:41:56 2022

@author: tiago
"""
# internal
from dataloader.dataloader import DataLoader
from utils.utils import Utils

# external
import tensorflow as tf
import numpy as np
from rdkit import Chem
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import csv

# Design of the reward functions 
# x_usp7 = np.linspace(0, 10, 100)
# y_usp7 = np.exp(x_usp7/3-1.1)
# plt.figure()
# plt.plot(x_usp7, y_usp7)
# plt.xlabel('pIC50 for USP7')
# plt.ylabel('Reward')

# # x_sas = np.linspace(0, 10, 100)
# y_sas = np.exp(-x_usp7/4 + 1.6)
# plt.figure()
# plt.plot(x_usp7, y_sas,'red')
# plt.xlabel('SAS')
# plt.ylabel('Reward')
# plt.show()


filepath = "sampled_mols_run_"
all_mols = []
all_pic50 = []
all_sas = []

for r in range(1,13):
    
    with open(filepath + str(r) + '.smi', 'r') as csvFile:
        reader = csv.reader(csvFile)
        
        it = iter(reader)
 
        for idx,row in enumerate(it):
            if idx > 3:
                mol = row[0]
                pic50 = row[1]
                sas = row[2]
                
                if mol not in all_mols:
                    all_mols.append(mol)
                    all_pic50.append(float(pic50))
                    all_sas.append(float(sas))
    
            
    
# identify non-dominated points
dominated_mols  = []
for idx in range(0,len(all_mols)):
    if (Utils.check(all_pic50,all_sas,all_pic50[idx],all_sas[idx])) == True:
        dominated_mols.append(idx)
        
all_pic50_nd = [element for idx,element in enumerate(all_pic50) if idx not in dominated_mols]
all_sas_nd = [element for idx,element in enumerate(all_sas) if idx not in dominated_mols]
all_mols_nd = [element for idx,element in enumerate(all_mols) if idx not in dominated_mols]

all_pic50_d = [element for idx,element in enumerate(all_pic50) if idx in dominated_mols]
all_sas_d = [element for idx,element in enumerate(all_sas) if idx in dominated_mols]
all_mols_d = [element for idx,element in enumerate(all_mols) if idx in dominated_mols]
 
# # # identify second non-dominated points
# second_dominated_mols  = []
# for idx in range(0,len(all_mols_d)):
#     if (Utils.check(all_pic50_d,all_sas_d,all_pic50_d[idx],all_sas_d[idx])) == True:
#         second_dominated_mols.append(idx)

# all_pic50_nd = [element for idx,element in enumerate(all_pic50_d) if idx not in second_dominated_mols]
# all_sas_nd = [element for idx,element in enumerate(all_sas_d) if idx not in second_dominated_mols]
# all_mols_nd = [element for idx,element in enumerate(all_mols_d) if idx not in second_dominated_mols]

# all_pic50_d = [element for idx,element in enumerate(all_pic50_d) if idx in second_dominated_mols]
# all_sas_d = [element for idx,element in enumerate(all_sas_d) if idx in second_dominated_mols]
# all_mols_d = [element for idx,element in enumerate(all_mols_d) if idx in second_dominated_mols]
        

plt.scatter(all_pic50_d , all_sas_d, color = 'b', label='Dominated', alpha=0.5)
plt.scatter(all_pic50_nd, all_sas_nd, color = 'r', label='Non-dominated', alpha=0.5)

plt.xlabel("pIC50 for the USP7 target")
plt.ylabel("SAS")
plt.title("Sampled molecules")
# plt.legend()
# plt.ylim(-1.45, -0.1)
# plt.xlim(-0.4,0.1)
plt.show()
     
all_data = pd.DataFrame()  
     
all_data['molecules'] = all_mols
all_data['pic50'] = all_pic50
all_data['SAS'] = all_sas

dominated = [idx in dominated_mols for idx in range(0,len(all_mols))]
all_data['dominated']  = dominated

all_data.to_pickle('pareto_generation_mols.pkl')