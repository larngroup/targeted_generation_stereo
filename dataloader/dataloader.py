# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""
# Internal
from model.generator import Generator  
from model.dnnQSAR_new import DnnQSAR_model

# External
from keras.models import Sequential
import csv
import openpyxl
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_generator(config,generator_type):
        """ Initializes and loads the weights of the specified trained 
            generator model

        Args
        ----------
            config (bunch): Configuration file
            generator_type (str): Indication if one want to load the unbiased 
                                  or the biased model 

        Returns
        -------
            generator_model (sequential): model with the trained weights
        """
                
        generator_model = Sequential()
        generator_model=Generator(config,True)
        
        path = ''
        if generator_type == 'biased':
            path = config.generator_biased_path
        elif generator_type == 'unbiased':
            path = config.generator_unbiased_path
        
        generator_model.model.load_weights(path)
        
        return generator_model
    
    @staticmethod
    def load_predictor(config):
        """ Initializes and loads biological affinity Predictor

        Args
        ----------
            config (bunch): Configuration file

        Returns
        -------
            predictor (object): The Predictor object that enables application 
                                of the model to perform predictions 
        """
        
        predictor_obj = DnnQSAR_model(config)
        
        return predictor_obj
    
    @staticmethod
    def load_promising_mols(config):
        """ Loads a set of pre-generated molecules

        Args
        ----------
            config (bunch): Configuration file

        Returns
        -------
            smiles_list (list): Set of promising hits previously generated
        """
        
        smiles = []
        pic50_values = []
        mw_values = []
        sas_values = []
        logp_values = []
        qed_values = []
        tpsa_values = []
        h_donors_values = []
        h_acceptors_values = []
        rotablebonds_values = []
        n_rings = []
        file_ids =[]
        df = pd.DataFrame()
            
    
        paths_old_pred = [config.path_promising_hits ,"generated/sample_mols_oldpred_rl.smi","generated/sample_mols_newpred_rl.smi"]
        
        for fp_id,fp in enumerate(paths_old_pred):
            with open(fp, 'r') as csvFile:
                reader = csv.reader(csvFile)
                
                it = iter(reader)
                # next(it, None)  # skip first item.    
                for idx,row in enumerate(it):
                        
                    try:
                        m = Chem.MolFromSmiles(row[0])
                        s = Chem.MolToSmiles(m)
                        if s not in smiles:
                            smiles.append(s)
                            print(fp_id)
                            if fp_id == 2:
                                pic50_values.append(float(row[1][1:-2]))
                            else:
                               pic50_values.append(float(row[1])) 
                            sas_values.append(float(row[2]))
                            mw_values.append(float(row[3]))
                            logp_values.append(float(row[4]))
                            qed_values.append(float(row[5]))
                            file_ids.append(fp_id)
                                    
                        
                            tpsa_values.append(round(Descriptors.TPSA(m),2))
                            h_donors_values.append(Chem.Lipinski.NumHDonors(m))
                            h_acceptors_values.append(Chem.Lipinski.NumHAcceptors(m))
                            rotablebonds_values.append(Chem.Lipinski.NumRotatableBonds(m))
                            n_rings.append(Chem.Lipinski.RingCount(m))
                    
                    except:
                        print('Invalid smiles')
            
       
        df['smiles'] = smiles
        df['pic50'] = pic50_values
        df['sas'] = sas_values
        df['mw'] = mw_values
        df['logp'] = logp_values
        df['qed'] = qed_values
        df['tpsa'] = tpsa_values 
        df['hdonors'] = h_donors_values
        df['hacceptors'] = h_acceptors_values 
        df['rotable_bonds'] = rotablebonds_values
        df['n_rings'] = n_rings 
        df['file_id']  = file_ids
        
        df_1_2 = df[(df['file_id']==0) | (df['file_id']==1) ]
        df_3 = df[df['file_id']==2]
        return df_1_2,df_3
    
    
        