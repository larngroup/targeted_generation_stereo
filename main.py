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
    
    
    conditional_generation.load_models()
    # conditional_generation.filter_promising_mols()
    conditional_generation.drawMols()
    # conditional_generation.select_best_stereoisomers()
    # conditional_generation.samples_generation()
    # conditional_generation.compare_models()
    # conditional_generation.policy_gradient()
    # gep_model.evaluate()
    # gep_model.save()
if __name__ == '__main__':
    run()
