# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:49:12 2021

@author: tiago
"""
# external
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from utils.sascorer_calculator import SAscore

class Utils:
    """Utils class"""
    
    def __init__(self):
        """ Definition of the SMILES vocabulary """
        
        self.table = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
                  'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
                  '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
                  '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
    
    @staticmethod
    def smilesDict(token_table):
        """ Computes the dictionary that makes the correspondence between 
        each token and the respective integer.

        Args
        ----------
            tokens (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenDict (dict): Dictionary mapping characters into integers
        """

        tokenDict = dict((token, i) for i, token in enumerate(token_table))
        return tokenDict
    
    @staticmethod
    def pad_seq(smiles,tokens,config):
        """ Performs the padding for each sampled SMILES.

        Args
        ----------
            smiles (str): SMILES string
            tokens (list): List of each possible symbol in the SMILES
            config (json): Configuration file

        Returns
        -------
            smiles (str): Padded SMILES string
        """
        
        if isinstance(smiles, str) == True:
            smiles = [smiles]
            
        maxLength = config.paddSize
    
        for i in range(0,len(smiles)):
            smiles[i] = 'G' + smiles[i] + 'E'
            if len(smiles[i]) < maxLength:
                smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))
        # print("Padded sequences: ", len(filtered_smiles))
        return smiles
    

    @staticmethod            
    def smiles2idx(smiles,tokenDict):
        """ Transforms each token into the respective integer.

        Args
        ----------
            smiles (str): Sampled SMILES string 
            tokenDict (dict): Dictionary mapping characters to integers 

        Returns
        -------
            newSmiles (str): Transformed smiles, with the characters 
                              replaced by the respective integers. 
        """   
        try:
            newSmiles =  np.zeros((len(smiles), len(smiles[0])))
        except:
            print(smiles)
        for i in range(0,len(smiles)):
            # print(i, ": ", smiles[i])
            for j in range(0,len(smiles[i])):
                
                try:
                    newSmiles[i,j] = tokenDict[smiles[i][j]]
                except:
                    value = tokenDict[smiles[i][j]]
        return newSmiles
        
    @staticmethod             
    def tokenize(config,smiles,token_table):
        """ Transforms the SMILES string into a list of tokens.

        Args
        ----------
            config (json): Configuration file
            smiles (str): Sampled SMILES string
            token_table (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenized (str):  SMILES string with individualized tokens.
        """           

        tokenized = []
        
        for idx,smile in enumerate(smiles):
            N = len(smile)
            i = 0
            j= 0
            tokens = []
            #print(idx,smile)
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            while (len(tokens) < config.paddSize):
                tokens.append(token_table[-1])
                
                
            tokenized.append(tokens)
    
        return tokenized

    @staticmethod             
    def idx2smi(model_output,tokenDict):
        """ Transforms model's predictions into SMILES

        Args
        ----------
            model_output (array): List with the autoencoder's predictions 
            tokenDict (dict): Dictionary mapping characters into integers

        Returns
        -------
            reconstructed_smiles (array): List with the reconstructed SMILES 
                                          obtained by transforming indexes into
                                          tokens. 
        """           

        key_list = list(tokenDict.keys())
        val_list = list(tokenDict.values())

        reconstructed_smiles =  []
        for i in range(0,len(model_output)):
            smi = []
            for j in range(0,len(model_output[i])):
                
                smi.append(key_list[val_list.index(model_output[i][j])])
                
            reconstructed_smiles.append(smi)
                
        return reconstructed_smiles
    
    @staticmethod
    def remove_padding(trajectory):  
        """ Function that removes the padding characters from the sampled 
            molecule

        Args
        ----------
            trajectory (str): Padded generated molecule

        Returns
        -------
            trajectory (str): SMILES string without the padding character
        """     
        
        if 'A' in trajectory:
            
            for idx,token in enumerate(trajectory):
                if token == 'A' :
                    idx_A = idx
                    new_trajectory = trajectory[1:idx_A-1]
                    break
        else:
            new_trajectory = trajectory[1:-1]
        return new_trajectory
    
    @staticmethod
    def reading_csv(config):
        """ This function loads the labels of the biological affinity dataset
        
        Args
        ----------
            config (json): configuration file
        
        Returns
        -------
            raw_labels (list): Returns the respective labels in a numpy array. 
        """
  
        raw_labels = []
            
        with open(config.file_path_predictor_data, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)
    #        next(it, None)  # skip first item.    
            for row in it:
               
        
                try:
                    raw_labels.append(float(row[1]))
                except:
                    pass
               
        return raw_labels
    
    @staticmethod 
    def get_reward_MO(predictor_obj,smile,memory_smiles):
        """ This function uses the predictor and the sampled SMILES string to 
        predict the numerical rewards regarding the evaluated properties.

        Args
        ----------
            predictor_affinity (object): Predictive model that accepts a trajectory
                                     and returns the respective prediction 
            smile (str): SMILES string of the sampled molecule
            memory_smiles (list): List of the last 30 generated molecules


        Returns
        -------
            rewards (list): Outputs the list of reward values for the evaluated 
                            properties
        """
    
        # pIC50 for USP7
        rewards = []
        list_smiles = [smile] 
        
        pred = predictor_obj.predict(list_smiles)[0][0]
        
        reward_affinity = np.exp(pred/3-1.1) # (pred/4-1)
        
        if pred < 5:
            rewards.append(0)
        else:
            rewards.append(reward_affinity)

        # SA score
        list_mol_sas = []
        list_mol_sas.append(Chem.MolFromSmiles(smile))
        sas_list = SAscore(list_mol_sas)
        sas = sas_list[0]
        rew_sas = np.exp(-sas/4 + 1.6)

        rewards.append(rew_sas)
       
        diversity = 1
        if len(memory_smiles)> 30:
            diversity = Utils.external_diversity(smile,memory_smiles)
            
        if diversity < 0.75:
            rew_div = 0.95
            print("\Alert: Similar compounds")
        else:
            rew_div = 1
            
        rewards.append(rew_div)
        return rewards,sas,pred

    @staticmethod
    def scalarization(rewards,scalarMode,weights,pred_range_pic50,pred_range_sas):
        """ Transforms the vector of two rewards into a unique reward value.
        
        Args
        ----------
            rewards (list): List of rewards of each property;
            scalarMode (str): String indicating the scalarization type;
            weights (list): List containing the weights indicating the importance 
                            of the each property;
            pred_range_pic50 (list): List with the max and min prediction values
                                of the reward to for the pic50 to normalize the
                                obtained reward (between 0 and 1).
            pred_range_sas (list): List with the max and min prediction values
                                of the reward to for the SAS to normalize the
                                obtained reward (between 0 and 1).

        Returns
        -------
            rescaled_reward (float): Scalarized reward
        """
        w_affinity = weights[0]
        w_sas = weights[1]
        
        rew_affinity = rewards[0]
        rew_sas = rewards[1]
        rew_div = rewards[2]
        
        max_affinity = pred_range_pic50[1]
        min_affinity = pred_range_pic50[0]
    
        max_sas = pred_range_sas[1]
        min_sas = pred_range_sas[0]
    
        rescaled_rew_sas = (rew_sas - min_sas )/(max_sas - min_sas)
        
        if rescaled_rew_sas < 0:
            rescaled_rew_sas = 0
        elif rescaled_rew_sas > 1:
            rescaled_rew_sas = 1
    
        rescaled_rew_affinity = (rew_affinity  - min_affinity )/(max_affinity -min_affinity)
        
        if rescaled_rew_affinity < 0:
            rescaled_rew_affinity = 0
        elif rescaled_rew_affinity > 1:
            rescaled_rew_affinity = 1
        
        if scalarMode == 'linear':
            return (w_affinity*rescaled_rew_affinity + w_sas*rescaled_rew_sas)*3*rew_div,rescaled_rew_sas,rescaled_rew_affinity
    
        elif scalarMode == 'chebyshev':

            dist_affinity = abs(rescaled_rew_affinity-1)*w_affinity
            dist_sas = abs(rescaled_rew_sas-1)*w_sas
            print("distance a2d: " + str(dist_affinity))
            print("distance sas: " + str(dist_sas))
            
            if dist_affinity > dist_sas:
                return rescaled_rew_affinity*3
            else:
                return rescaled_rew_sas*3
    
    @staticmethod 
    def external_diversity(set_A,set_B):
        """ Computes the Tanimoto external diversity between two sets
        of molecules

        Args
        ----------
            set_A (list): Set of molecules in the form of SMILES notation
            set_B (list): Set of molecules in the form of SMILES notation


        Returns
        -------
            td (float): Outputs a number between 0 and 1 indicating the Tanimoto
                        distance.
        """

        td = 0
        set_A = [set_A]
        fps_A = []
        for i, row in enumerate(set_A):
            try:
                mol = Chem.MolFromSmiles(row)
                fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!')
        if set_B == None:
            for ii in range(len(fps_A)):
                for xx in range(len(fps_A)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                    td += ts          
          
            td = td/len(fps_A)**2
        else:
            fps_B = []
            for j, row in enumerate(set_B):
                try:
                    mol = Chem.MolFromSmiles(row)
                    fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
                except:
                    print('ERROR: Invalid SMILES!') 
            
            
            for jj in range(len(fps_A)):
                for xx in range(len(fps_B)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                    td += ts
            
            td = td / (len(fps_A)*len(fps_B))
        print("Tanimoto distance: " + str(td))  
        return td
    
    def smiles2mol(smiles_list):
        """
        Function that converts a list of SMILES strings to a list of RDKit molecules 
        Parameters
        ----------
        smiles: List of SMILES strings
        ----------
        Returns list of molecules objects 
        """
        mol_list = []
        if isinstance(smiles_list,str):
            mol = Chem.MolFromSmiles(smiles_list, sanitize=True)
            mol_list.append(mol)
        else:
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                mol_list.append(mol)
        return mol_list
                
    def denormalization(predictions,data):
        """
        This function implements the denormalization step.
        ----------
        predictions: Output from the model
        data: original labels to extract q1 and q3 values
        
        Returns
        -------
        Returns the denormalized predictions.
        """
        q1_train = np.percentile(data, 5)
        q3_train = np.percentile(data, 90)
        
        for l in range(len(predictions)):

            for c in range(len(predictions[0])):
                predictions[l,c] = (q3_train - q1_train) * predictions[l,c] + q1_train

      
        return predictions

    @staticmethod 
    def padding_one_hot(smiles,tokens): 
        """ Performs the padding of the sampled molecule represented in OHE
        Args
        ----------
            smiles (str): Sampled molecule in the form of OHE;
            tokens (list): List of tokens that can constitute the molecules   

        Returns
        -------
            smiles (str): Padded sequence
        """

        smiles = smiles[0,:,:]
        maxlen = 65
        idx = tokens.index('A')
        padding_vector = np.zeros((1,43))
        padding_vector[0,idx] = 1
    
        while len(smiles) < maxlen:
            smiles = np.vstack([smiles,padding_vector])
                
        return smiles
    
   

    
    def plot_training_progress(training_rewards,losses_generator,training_pic50,training_sas, rewards_pic50, rewards_sas,scaled_rewards_pic50,scaled_rewards_sas):
        """ Plots the evolution of the rewards and loss throughout the 
        training process.
        Args
        ----------
            training_rewards (list): List of the combined rewards for each 
                                     sampled batch of molecules;
            losses_generator (list): List of the computed losses throughout the 
                                     training process;
            training_pic50 (list): List of the pIC50 values for each sampled 
                                     batch of molecules;
            training_sas (list): List of the SAS values for each sampled 
                                     batch of molecules;
            rewards_pic50 (list): List of the rewards for the pIC50 property
                                  for each sampled batch of molecules;
            rewards_sas (list): List of the rewards for the SAS property
                                  for each sampled batch of molecules;

        Returns
        -------
            Plot
        """

        plt.plot(training_rewards)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards')
        plt.show()
        
        plt.plot(training_pic50)
        plt.xlabel('Training iterations')
        plt.ylabel('Average pIC50')
        plt.show()

        plt.plot(training_sas)
        plt.xlabel('Training iterations')
        plt.ylabel('Average SA score')
        plt.show()
        
        plt.plot(losses_generator)
        plt.xlabel('Training iterations')
        plt.ylabel('Average losses PGA')
        plt.show()
        
        plt.plot(rewards_pic50)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards pic50')
        plt.show()
        
        plt.plot(rewards_sas)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards sas')
        plt.show()
        
        plt.plot(scaled_rewards_pic50)
        plt.xlabel('Training iterations')
        plt.ylabel('Scaled rewards pic50')
        plt.show()
        
        plt.plot(scaled_rewards_sas)
        plt.xlabel('Training iterations')
        plt.ylabel('Scaled rewards sas')
        plt.show()

                
    def moving_average(previous_values, new_value, ma_window_size=10): 
        """
        This function performs a simple moving average between the previous 9 and the
        last one reward value obtained.
        ----------
        previous_values: list with previous values 
        new_value: new value to append, to compute the average with the last ten 
                   elements
        
        Returns
        -------
        Outputs the average of the last 10 elements 
        """
        value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
        value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
        return value_ma
                
        
    def compute_thresh(rewards,thresh_set):
        """
        Function that computes the thresholds to choose which Generator will be
        used during the generation step, based on the evolution of the reward values.
        Parameters
        ----------
        rewards: Last 3 reward values obtained from the RL method
        thresh_set: Integer that indicates the threshold set to be used
        Returns
        -------
        This function returns a threshold depending on the recent evolution of the
        reward. If the reward is increasing the threshold will be lower and vice versa.
        """
        reward_t_2 = rewards[0]
        reward_t_1 = rewards[1]
        reward_t = rewards[2]
        q_t_1 = reward_t_2/reward_t_1
        q_t = reward_t_1/reward_t
        
        if thresh_set == 1:
            thresholds_set = [0.15,0.3,0.2]
        elif thresh_set == 2:
            thresholds_set = [0.05,0.2,0.1] 
        #        thresholds_set = [0,0,0] 
        
        threshold = 0
        if q_t_1 < 1 and q_t < 1:
            threshold = thresholds_set[0]
        elif q_t_1 > 1 and q_t > 1:
            threshold = thresholds_set[1]
        else:
            threshold = thresholds_set[2]
        
        return threshold

    def serialize_model(generator_biased,config,pol):
        """
        Takes the Generator and saves its parameters to disk
        Parameters
        ----------
        generator_biased: Generator object
        config: configuration file
        pol: current training iteration              
        """
        
        generator_biased.model.save('models//generator//biased_generator.hdf5')
        
        # model_json = generator_biased.model.to_json()
        # with open(config.model_name_biased + "_" +str(pol)+".json", "w") as json_file:
        #     json_file.write(model_json)
            
        # # serialize weights to HDF5
        # generator_biased.model.save_weights(config.model_name_biased + '_' +str(pol)+".h5")
        # print("Updated model saved to disk")
        
    def canonical_smiles(smiles,sanitize=True, throw_warning=False):
        """
        Takes list of generated SMILES strings and returns the list of valid SMILES.
        Parameters
        ----------
        smiles: List of SMILES strings to validate
        sanitize: bool (default True)
            parameter specifying whether to sanitize SMILES or not.
                For definition of sanitized SMILES check
                http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        throw_warning: bool (default False)
            parameter specifying whether warnings will be thrown if a SMILES is
            invalid
        Returns
        -------
        new_smiles: list of valid SMILES (if it is valid and has <60 characters)
        and NaNs if SMILES string is invalid
        valid: number of valid smiles, regardless of the its size
            
        """
        new_smiles = []
        valid = 0
        for sm in smiles:
            try:
                mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
                s = Chem.MolToSmiles(mol)
                
                if len(s) <= 75:
                    new_smiles.append(s)       
                else:
                    new_smiles.append('')
                valid = valid + 1 
            except:
                new_smiles.append('')
        return new_smiles,valid

    def plot_hist(prediction, n_to_generate,valid,property_identifier):
        """
        Function that plots the predictions's distribution of the generated SMILES 
        strings
        Parameters
        ----------
        prediction: list with the desired property predictions.
        n_to_generate: number of generated SMILES.
        valid: number of valid smiles, regardless of the its size.
        property_identifier: String identifying the property 
        Returns
        ----------
        Float indicating the percentage of valid generated molecules
        """
        prediction = np.array(prediction,dtype='float64').reshape((-1,))
        x_label = ''
        plot_title = '' 
        
        print("\n\nProportion of valid SMILES:", valid/n_to_generate)
        
        if property_identifier == "usp7":
            print("Max of pIC50: ", np.max(prediction))
            print("Mean of pIC50: ", np.mean(prediction))
            print("Std of pIC50: ", np.std(prediction))
            print("Min of pIC50: ", np.min(prediction))
            x_label = "Predicted pIC50"
            plot_title = "Distribution of predicted pIC50 for generated molecules"
            
        elif property_identifier == "sas":
            print("\n\nMax SA score: ", np.max(prediction))
            print("Mean SA score: ", np.mean(prediction))
            print("Std SA score: ", np.std(prediction))
            print("Min SA score: ", np.min(prediction))
            x_label = "Calculated SA score"
            plot_title = "Distribution of SA score for generated molecules"
        elif property_identifier == "qed":
            print("Max QED: ", np.max(prediction))
            print("Mean QED: ", np.mean(prediction))
            print("Min QED: ", np.min(prediction))
            x_label = "Calculated QED"
            plot_title = "Distribution of QED for generated molecules"  
            
        elif property_identifier == "logp":
            percentage_in_threshold = np.sum((prediction >= 0.0) & 
                                         (prediction <= 5.0))/len(prediction)
            print("Percentage of predictions within drug-like region:", percentage_in_threshold)
            print("Average of log_P: ", np.mean(prediction))
            print("Median of log_P: ", np.median(prediction))
            plt.axvline(x=0.0)
            plt.axvline(x=5.0)
            x_label = "Predicted LogP"
            plot_title = "Distribution of predicted LogP for generated molecules"
            
        sns.axes_style("darkgrid")
        ax = sns.kdeplot(prediction, shade=True,color = 'g')
        ax.set(xlabel=x_label,
               title=plot_title)
        plt.show()
        return (valid/n_to_generate)*100
    
    def pad_seq_pred(smiles,tokens,config):
        """ Performs the padding for each sampled SMILES.

        Args
        ----------
            smiles (str): SMILES string
            tokens (list): List of each possible symbol in the SMILES
            config (json): Configuration file

        Returns
        -------
            smiles (str): Padded SMILES string
        """
        
        if isinstance(smiles, str) == True:
            smiles = [smiles]
            
        maxLength = config.paddSize
    
        for i in range(0,len(smiles)):
            smiles[i] = 'G' + smiles[i] + 'E'
            if len(smiles[i]) < maxLength:
                smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))
        # print("Padded sequences: ", len(filtered_smiles))
        return smiles
    
    def tokenize_pred(config,smiles,token_table):
        """ Transforms the SMILES string into a list of tokens.

        Args
        ----------
            config (json): Configuration file
            smiles (str): Sampled SMILES string
            token_table (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenized (str):  SMILES string with individualized tokens.
        """           

        tokenized = []
        
        for idx,smile in enumerate(smiles):
            N = len(smile)
            i = 0
            j= 0
            tokens = []
            # print(idx,smile)
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            while (len(tokens) < config.paddSize):
                tokens.append(token_table[-1])
                

            tokenized.append(tokens)
    
        return tokenized
    
    def plot_hist_both(prediction_usp7_unb,prediction_usp7_b,prediction_sas_unb,prediction_sas_b, n_to_generate,valid_unb,valid_b):
        """
        Function that plots the predictions's distribution of the generated SMILES 
        strings, obtained by the unbiased and biased generators.
        Parameters
        ----------
        prediction_usp7_unb: list with the usp7 affinity predictions of unbiased 
                        generator.
        prediction_usp7_b: list with the usp7 affinity predictions of biased generator.
        prediction_sas_unb: list with the sas predictions of unbiased 
                        generator.
        prediction_sas_b: list with the sas predictions of biased generator.
        n_to_generate: number of generated molecules.
        valid_unb: number of valid molecules of the unbiased generator
        valid_b: number of valid smiles of the biased generator

        Returns
        ----------
        This functions returns the difference between the averages of the predicted
        properties and the % of valid SMILES
        """
        prediction_usp7_unb = np.array(prediction_usp7_unb)
        prediction_usp7_b= np.array(prediction_usp7_b)
        
        prediction_sas_unb = np.array(prediction_sas_unb)
        prediction_sas_b= np.array(prediction_sas_b)
        
        print("\nProportion of valid SMILES (UNB,B):", valid_unb/n_to_generate,valid_b/n_to_generate )
  
        legend_usp7_unb = 'Unbiased pIC50 values'
        legend_usp7_b = 'Biased pIC50 values'
        print("\n\nMax of pIC50: (UNB,B)", np.max(prediction_usp7_unb),np.max(prediction_usp7_b))
        print("Mean of pIC50: (UNB,B)", np.mean(prediction_usp7_unb),np.mean(prediction_usp7_b))
        print("Min of pIC50: (UNB,B)", np.min(prediction_usp7_unb),np.min(prediction_usp7_b))
    
        label_usp7 = 'Predicted pIC50'
        plot_title_usp7 = 'Distribution of predicted pIC50 for generated molecules'
            
  
        legend_sas_unb = 'Unbiased'
        legend_sas_b = 'Biased'
        print("\n\nMax of SA score: (UNB,B)", np.max(prediction_sas_unb),np.max(prediction_sas_b))
        print("Mean of SA score: (UNB,B)", np.mean(prediction_sas_unb),np.mean(prediction_sas_b))
        print("Min of SA score: (UNB,B)", np.min(prediction_sas_unb),np.min(prediction_sas_b))
    
        label_sas = 'Predicted SA score'
        plot_title_sas = 'Distribution of SA score values for generated molecules'  
     
        sns.axes_style("darkgrid")
        v1_usp7 = pd.Series(np.reshape(prediction_usp7_unb,[len(prediction_usp7_unb),]), name=legend_usp7_unb)
        v2_usp7 = pd.Series(np.reshape(prediction_usp7_b,[len(prediction_usp7_b),]), name=legend_usp7_b)
               
        ax = sns.kdeplot(v1_usp7, shade=True,color='b',label=legend_usp7_unb)
        sns.kdeplot(v2_usp7, shade=True,color='r',label =legend_usp7_b )
    
        ax.set(xlabel=label_usp7, 
               title=plot_title_usp7)
        # plt.legend()
        plt.show()
        
        v1_sas = pd.Series(prediction_sas_unb, name=legend_sas_unb)
        v2_sas = pd.Series(prediction_sas_b, name=legend_sas_b)
               
        ax = sns.kdeplot(v1_sas, shade=True,color='b',label=legend_sas_unb)
        sns.kdeplot(v2_sas, shade=True,color='r',label =legend_sas_b )
    
        ax.set(xlabel=label_sas, 
               title=plot_title_sas)
        # plt.legend()
        plt.show()
        
    
    def rmse(y_true, y_pred):
        """
        This function implements the root mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the rmse metric to evaluate regressions
        """
        from keras import backend
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def mse(y_true, y_pred):
        """
        This function implements the mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the mse metric to evaluate regressions
        """
        from keras import backend
        return backend.mean(backend.square(y_pred - y_true), axis=-1)

    def r_square(y_true, y_pred):
        """
        This function implements the coefficient of determination (R^2) measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the R^2 metric to evaluate regressions
        """
        from keras import backend as K
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))


    #concordance correlation coeﬃcient (CCC)
    def ccc(y_true,y_pred):
        """
        This function implements the concordance correlation coefficient (ccc)
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the ccc measure that is more suitable to evaluate regressions.
        """
        from keras import backend as K
        num = 2*K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred)))
        den = K.sum(K.square(y_true-K.mean(y_true))) + K.sum(K.square(y_pred-K.mean(y_pred))) + K.int_shape(y_pred)[-1]*K.square(K.mean(y_true)-K.mean(y_pred))
        return num/den

    def update_weights(scaled_rewards_rges,scaled_rewards_target,weights):
        
        mean_rges_previous = np.mean(scaled_rewards_rges[-5:-1])
        mean_rges_current = scaled_rewards_rges[-1:][0]
        
        mean_target_previous = np.mean(scaled_rewards_target[-5:-1])
        mean_target_current = scaled_rewards_target[-1:][0]
        
        growth_rges = (mean_rges_current - mean_rges_previous)/mean_rges_previous
        
        growth_target = (mean_target_current - mean_target_previous)/mean_target_previous
        
        if mean_rges_current*weights[0] > mean_target_current*weights[1] and growth_target < 0.01:
            weights[0] = weights[0] - 0.05
            weights[1] = weights[1] + 0.05
        elif mean_rges_current*weights[0] < mean_target_current*weights[1] and growth_rges < 0.01:
            weights[0] = weights[0] + 0.05
            weights[1] = weights[1] - 0.05
            
        print(weights) 
        
        return weights
    
    def check(list1,list2, val1,val2):
      
        # traverse in the list
        for idx in range(0,len(list1)):

            if list1[idx] > val1 and list2[idx] < val2:
                return True 
        return False

    def moving_average_new(raw_values,window_size):
        
        arr  = list(raw_values)
       
          
        i = 0
        # Initialize an empty list to store moving averages
        moving_averages = []
          
        # Loop through the array to consider
        # every window of size 3
        while i < len(arr) - window_size + 1:
            
            # Store elements from i to i+window_size
            # in list to get the current window
            window = arr[i : i + window_size]
          
            # Calculate the average of current window
            window_average = round(sum(window) / window_size, 2)
              
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
              
            # Shift window to right by one position
            i += 1
        
        return moving_averages