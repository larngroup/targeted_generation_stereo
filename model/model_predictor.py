# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:27:17 2022

@author: tiago
"""
import tensorflow as tf
from tensorflow.keras.layers import GRU
from model.attention import Attention

class Build_model(tf.keras.Model):
    """
    Predictor class model with the definition of the parameters and architecture
    """
    
    def __init__(self, config):
        super(Build_model, self).__init__()

        self.config = config
        self.inp_dimension = self.config.paddSize
        self.token_len = 47
        self.embedding_dim = 256
        self.bidirectional_units = 128
        self.dropout = 0.2
        self.rnn_units = 128
        
        self.input_layer = tf.keras.layers.Input(shape=(None,self.inp_dimension))
        self.embedding_layer = tf.keras.layers.Embedding(self.token_len,self.embedding_dim,input_length = self.inp_dimension)
        self.bidirectional_layer = tf.keras.layers.Bidirectional(GRU(self.bidirectional_units, dropout=self.dropout, return_sequences=True)) 
        self.rnn_layer = GRU(self.rnn_units, dropout=self.dropout, return_sequences=True)
        self.attention_layer = Attention()
        self.dense_layer = tf.keras.layers.Dense(1,activation='linear') 

        
    def call(self, sequence, training=True):

        # input_out = self.input_layer(sequence)
        embedding_out = self.embedding_layer(sequence)
        bidirectional_out = self.bidirectional_layer(embedding_out)
        rnn_out = self.rnn_layer(bidirectional_out)
        attention_out = self.attention_layer(rnn_out)
        pred_out = self.dense_layer(attention_out)
        
        return pred_out