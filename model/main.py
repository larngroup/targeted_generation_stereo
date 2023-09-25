# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:13:50 2020

@author: bjpsa
"""

import numpy as np
from Generator import Generator
from Vocabulary import Vocabulary as vocab1
from Vocabulary2 import Vocabulary as vocab2
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
import os
import csv
from time import process_time 
from functools import reduce
from tqdm import tqdm
import pickle

# tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)



def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t*1000,), 1000, 60, 60])

if __name__ == '__main__':

    
    headers = ['model', 'Valid', 'unique', 'time']
    
    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    

    curr_path = os.getcwd()
    print(curr_path)
    

    nr_smiles = 10000
    nr_layers = 2
    units = 512
    drop_rate = 0.2
    epochs = 1 #16
    batch_size = 16
    optimizer = 'RMSprop' #or 'adam', 'adam_clip', 'SGD'
    encode = 'embedding' #or 'OHE'
    dataset = 'ChEMBL'   #or 'biogenic'
    sampling_T = 0.75
    learning_rate = None
    activation = 'softmax'
   

    vocab = vocab2('Vocab.txt')
    path = './datasets/ChEMBL_filtered.txt'
    
    
    f_string = ''
    with open(path) as f:
        i = 0
        for line in f:
            #print(line)
            if len(line) < 98:
                f_string = f_string + line
                i+=1
            if(i>=nr_smiles):
              break
    smiles = f_string.split('\n')[:-1]
    
    
    smiles_padded = vocab.pad_seq(smiles)

    # Compute the dictionary that makes the correspondence between each token and unique integers
    tokenDict = vocab.smilesDict()
   
   	# Tokenize - transform the SMILES strings into lists of tokens 
    tokens = vocab.tokenize(smiles_padded)   

   # Transforms each token to the respective integer, according to the previously computed dictionary
    smiles_int = vocab.smiles2idx(tokens,tokenDict)
               
    dataX = smiles_int
    dataY = vocab.get_target(dataX, encode,tokenDict)
    # # Reshape
    data_X = np.reshape(dataX, (nr_smiles, 100, 1))  #(1000, 100, 1)
    #print(data_X.shape)
    data_Y = np.reshape(dataY, (nr_smiles,100))           # when using sparse_categorical_crossentropy
    #data_Y = to_categorical(dataY, num_classes = vocab.vocab_size)  # when using categorical cross entropy

    name = 'LSTM - '+ str(nr_layers) + " Layer"

     
    start_train_t = process_time()
    generator = Generator(curr_path, dataset, encode, nr_smiles, vocab, 100, nr_layers, units, learning_rate, drop_rate, activation, epochs, batch_size, optimizer,sampling_T)
    results, last_epoch = generator.fit_model(data_X, data_Y)
    # generator.load_model("Exp8_Temperature_embedding_10000_LSTM-2-512-1-16-0.2-RMSprop-256-0.75\\weights-improvement-01-0.6843.hdf5")
    end_train_t = process_time()
    train_time = end_train_t - start_train_t
                                    
    path_generator = generator.path
    with open(path_generator + 'results_'+optimizer+'.txt', "wb") as fp:
        pickle.dump(results.history['loss'], fp)

    # results_opt.append(results.history['loss'])
    #Generate new molecules
    
    #start_gen = process_time()
    #generating molecules in batches
    for _ in range(10):
        n_generated = 100
        new = generator.generate(vocab.char_to_int['G'], n_generated, vocab.char_to_int['A'])
        
        if encode == 'embedding':
            generated = vocab.decode(new) #emb
        elif encode == 'OHE':
            generated2 = []
            for l in range(len(new)):
                sm = []
                for k in new[l]:
                    idx = np.argmax(k)
                    sm.append(idx)
                generated2.append(sm)
            generated = vocab.decode(generated2)
            
        with open(path_generator + 'generatedMol_T_'+str(sampling_T)+'.txt', 'a') as f:
            for mol in generated:
                f.write(mol+'\n')

    #end_gen = process_time()
    #t_generate = end_gen-start_gen  
    #print(secondsToStr(t_generate))   
    #print(t_generate)  

    ## loading the file with generated molecules
    f_mol =  ''
    #with open(path_generator + 'generatedMol.txt', 'r') as f:
    with open(path_generator + 'generatedMol_T_'+str(sampling_T)+'.txt', 'r') as f:
        for line in f:
            f_mol = f_mol + line
    generated_smiles = f_mol.split('\n')[:-1]

    # generated = vocab.decode(new)
    _, perc = validity(generated_smiles)
    uniq = uniqueness(generated_smiles)
    #int_div = diversity(generated_smiles)
    #ext_div = diversity(generated_smiles, smiles)
    

    
    
    row = ['LSTM-'+str(layer)+'layers_'+ optimizer,t, perc, uniq, secondsToStr(train_time)]
    
    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
                           
  
                                

