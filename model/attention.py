# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""

from keras.layers import Layer
import keras.backend as K
import matplotlib.pyplot as plt
from utils.utils import Utils

class Attention(Layer):
    """Attention mechanism class"""

    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(Attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)


        # at_np = at.numpy()
        # ax_sq = at_np[1,:,0] # best
        
        # window_size = 3
        # moving_avg = Utils.moving_average_new(ax_sq,window_size)
        # moving_avg.insert(0, moving_avg[0])
        # moving_avg.insert(0, moving_avg[0])
        
        # plt_1 = plt.figure(figsize=(15,7))
        # seq = 'O=C(NC(Cc1ccccc1)C)CC(CC(=O)Nc1c(OC)ccc(C(=O)NCc2ccc(C(=O)NN)cc2)c1)(C)C'
        # # seq = 'FC(F)(F)c1ccc(-c2nc(Nc3c(C)ccc(C(=O)Nc4ccc(C(C)C)cc4)c3)ncc2)cc1'
        
        # seq = seq + 'A'*(75 - len(seq))
        # plt.plot(moving_avg,linestyle='dashed')
        # ax = plt.gca()
        # ax.set_xticks(range(len(seq)))
        # ax.set_xticklabels(seq)
        # plt.xlabel('Training iterations')
        # plt.ylabel('Attention weights')
        # plt.show()

        
        
        
        
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from matplotlib.collections import LineCollection
        # from matplotlib.colors import ListedColormap, BoundaryNorm
        
        # x = np.linspace(0, 3 * np.pi, 500)
        # y = np.sin(x)
        # z = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
        
        # # Create a colormap for red, green and blue and a norm to color
        # # f' < -0.5 red, f' > 0.5 blue, and the rest green
        # cmap = ListedColormap(['r', 'g', 'b'])
        # norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
        
        # # Create a set of line segments so that we can color them individually
        # # This creates the points as a N x 1 x 2 array so that we can stack points
        # # together easily to get the segments. The segments array for line collection
        # # needs to be numlines x points per line x 2 (x and y)
        # points = np.array([x, y]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # # Create the line collection object, setting the colormapping parameters.
        # # Have to set the actual values used for colormapping separately.
        # lc = LineCollection(segments, cmap=cmap, norm=norm)
        # lc.set_array(z)
        # lc.set_linewidth(3)
        
        # fig1 = plt.figure()
        # plt.gca().add_collection(lc)
        # plt.xlim(x.min(), x.max())
        # plt.ylim(-1.1, 1.1)        
        
        
        
        output=x*at
        return K.sum(output,axis=1)
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(Attention,self).get_config()