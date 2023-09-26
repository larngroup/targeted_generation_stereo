# -*- coding: utf-8 -*-
# Internal 
from utils.utils import Utils
from model.model_predictor import Build_model

# External 
import tensorflow as tf
import numpy as np

class BaseModel(object):
    def __init__(self,config):
        """
        This class implements the DNN-based QSAR model.
        """
        
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
          'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
          '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
          '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
        
        self.labels = Utils.reading_csv(config)
        self.config = config
       
class DnnQSAR_model(BaseModel):
    
    def __init__(self,config):
        super(DnnQSAR_model, self).__init__(config)
        
        
        self.predictor = Build_model(self.config)
        
        sequence_in = tf.constant([list(np.ones(100))])
        prediction_test = self.predictor(sequence_in)
        self.predictor.load_weights(self.config.predictor_path)
        
             
    def predict(self, smiles_original):
        """
        This function performs the prediction of the USP7 pIC50 for the input 
        molecules
        Parameters
        ----------
        smiles_original: List of SMILES strings to perform the prediction      
        Returns
        -------
        Before do the prediction, this function performs the SMILES' padding 
        and tokenization. It also performs the denormalization of the prediction
        """
        
        smiles = smiles_original.copy() 
        smiles = [s for s in smiles if len(s)<=98]
        smiles_padded = Utils.pad_seq_pred(smiles,self.tokens,self.config)
        
        d = Utils.smilesDict(self.tokens)
  
        tokens = Utils.tokenize_pred(self.config,smiles_padded,self.tokens)
                          
        smiles_int = Utils.smiles2idx(tokens,d)
        
        prediction = self.predictor.predict(smiles_int)

            
        prediction = Utils.denormalization(prediction,self.labels)
                
        # prediction = np.mean(prediction, axis = 0)
        # prediction = list(np.array(prediction,dtype="float64")) 

     
        return prediction
        
