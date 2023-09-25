# -*- coding: utf-8 -*-
# Internal 
from model.attention import Attention
from utils.utils import Utils

# External 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input,GRU,Bidirectional, Dropout
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras import Model
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

class BaseModel(object):
    def __init__(self,config):
        """
        This class implements the DNN-based QSAR model.
        """
    
        self.model = None
        # RNN parameters
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.n_units = 128
        self.embedding_dim = 64
        self.epochs = 115
        self.activation_rnn = "relu"
        self.batch_size = 16
        self.rnn = "lstm"
        self.input_length = 100
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.loss_criterium = "mean_squared_error"
        
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
          'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
          '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
          '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
        
                
        self.opt = Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, amsgrad=False)
        
        self.config = config
        self.labels = Utils.reading_csv(config)

class DnnQSAR_model(BaseModel):
    
    def __init__(self,config):
        super(DnnQSAR_model, self).__init__(config)
    
        self.build_model(len(self.tokens))
        
        self.loaded_models = []
      
        for i in range(5):
            
            loaded_model = self.model
                # load weights into new model
            loaded_model.load_weights(self.config.predictor_path +str(i+1)+".hdf5")      

            
            print("Model " + str(i) + " loaded from disk!")
            self.loaded_models.append(loaded_model)
        
    
    def build_model(self, n_table):
        """
        Depending on the descriptor type, it implements two different architectures
        For SMILES strings, we use a RNN. For the ECFP vectors, we use FCNN with 3 FC layers
        """
        
        self.n_table = n_table
        
        # self.model = Sequential()
        # self.model.add(Input(shape=(self.input_length,)))
        # self.model.add(Embedding(n_table,self.embedding_dim,input_length = self.input_length))
        
        # if self.rnn == 'lstm':
        #     self.model.add(LSTM(self.n_units, dropout=self.dropout,return_sequences=True))
        #     self.model.add(LSTM(self.n_units, dropout=self.dropout))
        # elif self.rnn == 'gru':
        #     self.model.add(GRU(256, dropout=0.2,input_shape=(None, 128,self.config.input_length),
        #                                            return_sequences=True))
        #     self.model.add(GRU(256, dropout=0.2))
            
        # # self.model.add(Dense(512,activation='relu')) 
        # self.model.add(Dense(1,activation='linear'))
        
        input_data = Input(shape=(self.input_length,), name = 'encoder_inputs')
  
        x = Embedding(n_table,64,input_length = self.input_length) (input_data)

        layer_bi_1 = Bidirectional(GRU(128, dropout=0.2, return_sequences=True))

        x_combined = layer_bi_1(x)
        
        x_att = Attention()(x_combined)
        
 
        # plt.plot(ax_sq)
        # ax = plt.gca()
        # ax.set_xticks(range(len(seq)))
        # ax.set_xticklabels(seq)
        # plt.xlabel('Training iterations')
        # plt.ylabel('Average rewards')
        # plt.show()

        # x_combined = Dense(256,activation = 'relu')(x_att)
        output = Dense(1,activation='linear') (x_att)

        self.model = Model(input_data, output)  

        self.model.summary()
        self.model.compile(loss=self.loss_criterium, optimizer = self.opt, metrics=[Utils.r_square,Utils.rmse,Utils.ccc])

        self.model.summary()
 
        # self.n_table = n_table

        # self.model = Sequential()
        # self.model.add(Input(shape=(self.input_length,)))
        # self.model.add(Embedding(n_table, self.n_units, input_length=self.input_length))

   
        # self.model.add(GRU(self.n_units, return_sequences=True, input_shape=(None,self.n_units,self.input_length),dropout = self.dropout))
        # self.model.add(GRU(self.n_units,dropout = self.dropout))

        # self.model.add(Dense(self.n_units, activation='relu'))
        # self.model.add(Dense(1, activation='linear'))
      
        # self.model.compile(loss=self.loss_criterium, optimizer = self.opt, metrics=[Utils.r_square,Utils.rmse,Utils.ccc])

    
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
        and tokenization. It also performs the denormalization step and compute 
        the mean value of the prediction of the 5 models.
        """
        
        smiles = smiles_original.copy() 
        smiles = [s for s in smiles if len(s)<=98]
        smiles_padded = Utils.pad_seq_pred(smiles,self.tokens,self.config)
        
        d = Utils.smilesDict(self.tokens)
  
        tokens = Utils.tokenize_pred(self.config,smiles_padded,self.tokens)
                          
        smiles_int = Utils.smiles2idx(tokens,d)
        
        prediction = []
            
        for m in range(len(self.loaded_models)):
            
            prediction.append(self.loaded_models[m].predict(smiles_int))

        prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
        
        prediction = Utils.denormalization(prediction,self.labels)
                
        prediction = np.mean(prediction, axis = 0)
    

     
        return prediction
        
