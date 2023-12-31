# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:05:42 2021

@author: tiago
"""
# internal
from model.generation import generation_process
from configs.configs import configs

# external
import warnings
warnings.filterwarnings('ignore')


config_file = 'configs\configGeneration.json' # Configuration file 

def run():
    """Loads and combines the required models into the dynamics of generating
    novel molecules"""
    
    # load configuration file
    cfg_file=configs.load_cfg(config_file)
     
    # Implementation of the generation dynamics
    conditional_generation = generation_process(cfg_file)
    
    # Build and load the DL models 
    conditional_generation.load_models()
    
    # Apply RL to optimize the Generator towards USP7 inhibitors
    conditional_generation.policy_gradient()
    conditional_generation.compare_models()
    
    # Generates and saves a set of molecules from the optimized model 
    conditional_generation.samples_generation()
    
    # Filters the most promising molecules according to their properties
    conditional_generation.filter_promising_mols()
    
    # Selects the best stereoisomers for each promising molecule
    conditional_generation.select_best_stereoisomers()
    
    # Draws the chemical 2D structure of promising hit compounds
    conditional_generation.drawMols()
    


    



if __name__ == '__main__':
    run()
