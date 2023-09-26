# -*- coding: utf-8 -*-

# External 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Generator:
    """
    Generator model class with the definition of the parameters and architecture
    """
    
    def __init__(self,config,unbiased):
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
                  'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
                  '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
                  '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
         
        self.config = config
        self.unbiased = unbiased
        self.model = None

        self.vocab_size = len(self.tokens)
        self.emb_dim = self.config.embb_generator
        self.max_len = self.config.paddSize
        
        self.n_layers = self.config.n_layers
        self.units = 512

        self.dropout_rate = self.config.dropout
        self.activation = self.config.activation_generator
        
        self.epochs = self.config.epochs_generator
        self.batch_size = self.config.batch_size_generator
        
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')

        self.build()
   
    def build(self):
        """
        Initializes the model
        """
        self.model = Sequential()


        self.model.add(Embedding(self.vocab_size, self.emb_dim, input_length = self.max_len))
        
        for i in range(self.n_layers): 
            self.model.add(LSTM(self.units, return_sequences=True))            
            if self.dropout_rate != 0:
                self.model.add(Dropout(self.dropout_rate))

        
        self.model.add(Dense(units = self.vocab_size, activation = self.activation))
        
        print(self.model.summary())
        
        if self.unbiased == True:
            self.model.compile(optimizer = self.optimizer, loss = 'sparse_categorical_crossentropy') #'mse' emb
        

    def load_model(self, path):
        """
        Loads the pre-trained model weights
        """
        self.model.load_weights(path)

    def fit_model(self, dataX, dataY):
        """
        Model pre-training step
        """

        filename="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"  
        early_stop = EarlyStopping(monitor = "loss", patience=5)
        path = self.path+F"{filename}"
        checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, mode = 'min') 
        callbacks_list = [checkpoint, early_stop]#, PlotLossesKerasTF()]
        results = self.model.fit(dataX, dataY, verbose = 1, epochs = self.epochs, batch_size = self.batch_size, shuffle = True, callbacks = callbacks_list)
        
        #plot
        fig, ax = plt.subplots()
        ax.plot(results.history['loss'])
        ax.set(xlabel='epochs', ylabel = 'loss')
        figure_path = self.path + "Loss_plot.png"
        fig.savefig(figure_path)
        #plt.show()
        last_epoch = early_stop.stopped_epoch
        
        return results, last_epoch
    
    
    def sample_with_temp(self, preds):
        """
        #samples an index from a probability array 'preds'
        preds: probabilities of choosing a character
        
        """
       # self.config.sampling_temp
        preds_ = np.log(preds).astype('float64')/0.8
        probs= np.exp(preds_)/np.sum(np.exp(preds_))
        #out = np.random.choice(len(preds), p = probs)
        
        # out_normal=np.argmax(np.random.multinomial(1,probs, 1))
        out_old = np.random.choice(len(probs), p=probs)

        return out_old
        
    
    def generate(self, numb):
        """
        Generates new SMILES strings, token by token

        Parameters
        ----------
        numb : int
            DESCRIPTION. number of SMILES strings to be generated

        Returns
        -------
        list_seq : TYPE list of list
            DESCRIPTION. A list where each entry is a tokenized SMILES 

        """

        list_seq = []
        list_ints = []
        

        start_idx = self.tokens.index('G')       
        end_idx = self.tokens.index('E')       
        for j in tqdm(range(numb)):
            list_ints = [start_idx]
            smi = 'G'
            #x = np.reshape(seq, (1, len(seq),1))
            
            for i in range(self.max_len-1):
                x = np.reshape(list_ints, (1, len(list_ints),1))
                preds = self.predict(x)
                
                #sample
                #index = np.argmax(preds[0][-1])
                #sample with T
                index = self.sample_with_temp(preds[0][-1])
                list_ints.append(index)
                smi +=self.tokens[index]
                if (index) == end_idx:
                    break
            list_seq.append(smi)

        return list_ints,list_seq
        
    def predict(self, input_x):
        preds = self.model.predict(input_x, verbose=1)
        return preds

    def smilesDict(self):
        """ Computes the dictionary that makes the correspondence between 
        each token and an given integer.

        Args
        ----------
            tokens (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenDict (dict): Dictionary mapping characters into integers
        """

        tokenDict = dict((token, i) for i, token in enumerate(self.tokens))
        return tokenDict
    
    def pad_seq(self,smiles):
        """ Performs the padding for each SMILE. To speed up the process, the
            molecules are previously filtered by their size.

        Args
        ----------
            smiles (list): SMILES strings with different sizes
            tokens (list): List of each possible symbol in the SMILES

        Returns
        -------
            smiles (list): List of padded smiles (with the same size)
            maxLength (int): Number that indicates the padding size
        """        
            
        for i in range(0,len(smiles)):
            smiles[i] = 'G' + smiles[i] + 'E'
            if len(smiles[i]) < 100:
                smiles[i] = smiles[i] + self.tokens[-1]*(100 - len(smiles[i]))
        # print("Padded sequences: ", len(filtered_smiles))
        return smiles    
    
    def tokenize(self,smiles):
        """ Transforms SMILES strings into a list of tokens.

        Args
        ----------
            config (json): Configuration file
            smiles (list): SMILES strings with different sizes
            token_table (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenized (list): List of SMILES with individualized tokens. The 
                              compounds are filtered by length, i.e., if it is
                              higher than the defined threshold, the compound
                              is discarded.
        """           

        tokenized = []
        
        for idx,smile in enumerate(smiles):
            N = len(smile)
            i = 0
            j= 0
            tokens = []
            # print(idx)
            while (i < N):
                for j in range(len(self.tokens)):
                    symbol = self.tokens[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            while (len(tokens) < 100):
                tokens.append(self.tokens[-1])
                
            tokenized.append(tokens)
        return tokenized    
        
    def smiles2idx(self,smiles,tokenDict):
        """ Transforms each token in the SMILES into the respective integer.

        Args
        ----------
            smiles (list): SMILES strings with different sizes
            tokenDict (dict): Dictionary mapping characters to integers 

        Returns
        -------
            newSmiles (list): List of transformed smiles, with the characters 
                              replaced by the numbers. 
        """   
        
        newSmiles =  np.zeros((len(smiles), len(smiles[0])))
        for i in range(0,len(smiles)):
            # print(i, ": ", smiles[i])
            for j in range(0,len(smiles[i])):
                
                try:
                    newSmiles[i,j] = tokenDict[smiles[i][j]]
                except:
                    value = tokenDict[smiles[i][j]]
        return newSmiles
    
         
    def get_target(self, dataX, smiles_dict):
         
          '''
          Creates the target for the input dataX

          Parameters
          ----------
          dataX : TYPE 
              DESCRIPTION.

          Returns
          -------
          dataY : TYPE equals dataX but with each entry shifted 1 timestep and with an appended 'A' (padding)
              DESCRIPTION.

          '''
             
          data_y = np.zeros([len(dataX),len(dataX[0])])
          
          for i in range(len(data_y)):
              for j in range(len(data_y[0])):
                  
                  if j < len(dataX[0])-1:
                      data_y[i,j] = dataX[i,j+1]
                  else:
                      data_y[i,j] = smiles_dict[self.tokens[-1]]

     
          return data_y
        

if __name__ == '__main__':
    pass
