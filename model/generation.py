    # -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader
from utils.utils import Utils
from model.Smiles_to_tokens import SmilesToTokens  
from model.predictSMILES import *
from model.generator import Generator  
from utils.sascorer_calculator import SAscore
from model.dnnQSAR import DnnQSAR_model

# external
import tensorflow as tf
import numpy as np
from rdkit import Chem
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


tf.config.experimental_run_functions_eagerly(True)

class generation_process(BaseModel):
    
    """Conditional Generation Object"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Load the table of possible tokens
        self.token_table = Utils().table 
        self.tokenDict = Utils.smilesDict(self.token_table)
        
        # Initialize optimizer and model
        self.adam = optimizers.Adam(clipvalue=4)
        self.generator_biased = Sequential()
        
        # Definition of the optimization parameters
        self.scalarMode = self.config.scalarMode
        self.pred_range_pic50 = [1.3,3.5]
        self.pred_range_sas = [1.25,3.4] 
        self.weights = [0.5,0.5]

    def load_models(self):
        """ Loads the Generator and pIC50 predictor against the USP7
        """      
        self.generator_unbiased = DataLoader().load_generator(self.config,'unbiased')
        self.predictor_usp7 = DnnQSAR_model(self.config)
    
    def custom_loss(self,aux_matrix):
        """ Computes the loss function to update the generator through the 
        policy gradient algorithm

        Args
        ----------
            aux_matrix (array): Auxiliary matrix to adjust the dimensions and 
                                padding when performing computations.

        Returns
        -------
            lossfunction (float): Value of the loss 
        """
        def lossfunction(y_true,y_pred):
            y_pred = tf.cast(y_pred, dtype='float64')
            y_true = tf.cast(y_true, dtype='float64')
            y_true = tf.reshape(y_true, (-1,))
           
            return (-1/self.config.batch_size)*K.sum(y_true*K.log(tf.math.reduce_sum(tf.multiply(tf.add(y_pred,10**-9),aux_matrix),[1,2])))
       
        return lossfunction

    def get_unbiased_model(self,aux_array):
        """ Builds the generator model pre-trained to generate valid molecules

        Args
        ----------
            aux_matrix (array): Auxiliary matrix to adjust the loss function
                                dimensions when performing computations.

        Returns
        -------
            generator_biased (model): Pre-trained Generator to be updated with
                                      the policy-gradient algorithm
        """
        
        self.generator_biased=Generator(self.config,False)
        self.generator_biased.model.compile(
                optimizer=self.adam,
                loss = self.custom_loss(aux_array))     
        self.generator_biased.model.load_weights(self.config.generator_unbiased_path)
        
        return self.generator_biased.model
        

    def policy_gradient(self, gamma=1):   
        """ Implements the policy gradient algorithm. """
         
        policy = 1
        cumulative_rewards = []

        # Initialize the variables that will contain the output of each prediction
        dimen = len(self.token_table)
        states = []
        
        pol_pic50 = []
        pol_sas = []
        pol_rewards_pic50 = []
        pol_rewards_sas = [] 
        pol_pic50_reward_scaled = []
        pol_sas_reward_scaled = []
        
        all_rewards = []
        losses_generator = []
        memory_smiles = []
        
        # Re-compile the model to adapt the loss function and optimizer to the RL problem
        self.generator_biased.model = self.get_unbiased_model(np.arange(47))
        
        for i in range(self.config.n_iterations):
        
            cur_reward = 0
            cur_pic50 = 0
            cur_sas = 0
            cur_reward_pic50 = 0 
            cur_reward_sas = 0  
            cur_reward_pic50_scaled = 0 
            cur_reward_sas_scaled = 0 
           
            aux_2 = np.zeros([100,47])
            inputs = np.zeros([100,1])
            
            ii = 0
            
            for m in range(self.config.batch_size):
                # Sampling new trajectory
                correct_mol = False
                
                while correct_mol != True:
                    trajectory_ints,trajectory = self.generator_biased.generate(1)
                    try:                     
                        seq = trajectory[0][1:-1]
                        if 'A' in seq: # A is the padding character
                            seq = Utils.remove_padding(trajectory) 

                        mol = Chem.MolFromSmiles(seq)
     
                        trajectory = 'G' + Chem.MolToSmiles(mol) + 'E'
                    
                        if len(memory_smiles) > 30:
                                memory_smiles.remove(memory_smiles[0])                                    
                        memory_smiles.append(seq)
                        
                        if len(trajectory) < self.config.paddSize:
                            correct_mol = True
                                               
                    except:
                        print("\nInvalid SMILES!")

                ii = 0
                     
                # Processing the sampled molecule         
                mol_padded = Utils.pad_seq(seq,self.token_table,self.config)
                tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
                trajectory_ints = remove_padding
                
                processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
            
                rewards,sas,pic50 = Utils.get_reward_MO(self.predictor_usp7,seq,memory_smiles)
            
                reward,rescaled_sas,rescaled_pic50 = Utils.scalarization(rewards,self.scalarMode,self.weights,self.pred_range_pic50,self.pred_range_sas)
            
                discounted_reward = reward
                cur_reward += reward
                cur_reward_pic50 += rewards[0]
                cur_reward_sas += rewards[1]
                cur_pic50 += pic50
                cur_sas += sas 
                cur_reward_pic50_scaled += rescaled_pic50
                cur_reward_sas_scaled += rescaled_sas
                
                # "Following" the trajectory and accumulating the loss
                inp_p = np.zeros([100,1])
                
                for p in range(1,len(trajectory_ints)):
            
                    states.append(discounted_reward)
                    inp_p[p-1,0] = processed_mol[0,p-1]
                    aux_2_matrix = np.zeros([100,47])
                    aux_2_matrix[p-1,int(processed_mol[0,p])] = 1
            
                    if ii == 0:
                        aux_2 = aux_2_matrix
                        inputs = np.copy(inp_p)
            
                    else:
                        inputs = np.dstack([inputs,inp_p])
                        aux_2 = np.dstack([aux_2,aux_2_matrix])
  
                    ii += 1
                
                inputs = np.moveaxis(inputs,-1,0)
                new_states = np.array(states)
                aux_2 = np.moveaxis(aux_2,-1,0)
    
                self.generator_biased.model.compile(optimizer = self.adam, loss = self.custom_loss(tf.convert_to_tensor(aux_2, dtype=tf.float64, name=None)))
               
                loss_generator = self.generator_biased.model.train_on_batch(inputs,new_states) # (inputs,targets) update the weights with a batch
            
                # Clear out variables
                states = []
                inputs = np.empty(0).reshape(0,0,dimen)
            
                cur_reward = cur_reward / self.config.batch_size
                cur_pic50 = cur_pic50 / self.config.batch_size
                cur_sas = cur_sas / self.config.batch_size
                cur_reward_pic50 = cur_reward_pic50 / self.config.batch_size
                cur_reward_sas = cur_reward_sas / self.config.batch_size
                cur_reward_pic50_scaled = cur_reward_pic50_scaled  / self.config.batch_size
                cur_reward_sas_scaled  = cur_reward_sas_scaled  / self.config.batch_size
            
                # serialize model to JSON
                Utils.serialize_model(self.generator_biased,self.config,policy)
            
                if len(all_rewards) > 2: # decide the threshold of the next generated batch 
                    self.config.threshold_greedy = Utils.compute_thresh(all_rewards[-3:],self.config.threshold_set)
             
                all_rewards.append(Utils.moving_average(all_rewards, cur_reward)) 
                pol_pic50.append(Utils.moving_average(pol_pic50, cur_pic50)) 
                pol_sas.append(Utils.moving_average(pol_sas, cur_sas))                   
                pol_rewards_pic50.append(Utils.moving_average(pol_rewards_pic50,cur_reward_pic50))
                pol_rewards_sas.append(Utils.moving_average(pol_rewards_sas,cur_reward_sas))
                pol_pic50_reward_scaled.append(Utils.moving_average(pol_pic50_reward_scaled, cur_reward_pic50_scaled))  
                pol_sas_reward_scaled.append(Utils.moving_average(pol_sas_reward_scaled, cur_reward_sas_scaled)) 
                losses_generator.append(Utils.moving_average(losses_generator, loss_generator))
            
                Utils.plot_training_progress(all_rewards,losses_generator,pol_pic50,pol_sas,pol_rewards_pic50,pol_rewards_sas,pol_pic50_reward_scaled,pol_sas_reward_scaled)
    
                if i%5==0 and i > 0:
                    self.weights = Utils.update_weights(pol_pic50_reward_scaled,pol_sas_reward_scaled,self.weights)
                    
            cumulative_rewards.append(np.mean(all_rewards[-10:]))
            policy+=1
    
        return cumulative_rewards
    
    def compare_models(self):
        """
        Samples molecules from the unbiased and biased Generators to 
        compare the efficiency of the optimization process 

        Returns
        -------
        Metrics of the sampled sets: pIC50, SAS, validity, uniqueness and Tanimoto 
        diversity
        """
        
        self.generator_unbiased = DataLoader().load_generator(self.config,'unbiased')
        self.generator_biased = DataLoader().load_generator(self.config,'biased')
    
        _,trajectory = self.generator_unbiased.generate(self.config.mols_to_generate)
        
        smiles_unbiased = [smile[1:-1] for smile in trajectory]
        
        sanitized_unb,valid_unb = Utils.canonical_smiles(smiles_unbiased, sanitize=True, throw_warning=False) # validar 

        smiles_sanitized_valid_unb = []
        
        for smi in sanitized_unb:
            if len(smi)>1:
                smiles_sanitized_valid_unb.append(smi)
        
        #prediction usp7 affinity
        prediction_usp7_unb = self.predictor_usp7.predict(smiles_sanitized_valid_unb)
        
        # prediction SA score
        mols_list_unb = Utils.smiles2mol(smiles_sanitized_valid_unb)
        prediction_sas_unb = SAscore(mols_list_unb)

        Utils.plot_hist(prediction_sas_unb,self.config.mols_to_generate,valid_unb,"sas")
        Utils.plot_hist(prediction_usp7_unb,self.config.mols_to_generate,valid_unb,"usp7")
         
        _,trajectory = self.generator_biased.generate(self.config.mols_to_generate)
        
        smiles_biased = [smile[1:-1] for smile in trajectory]
        
        sanitized_b,valid_b = Utils.canonical_smiles(smiles_biased, sanitize=True, throw_warning=False) # validar 
        
        smiles_sanitized_valid_b = []
        
        for smi in sanitized_b:
            if len(smi)>1:
                smiles_sanitized_valid_b.append(smi)
                       
        #prediction usp7 affinity
        prediction_usp7_b = self.predictor_usp7.predict(smiles_sanitized_valid_b)
    
        # prediction SA score
        mols_list_b = Utils.smiles2mol(smiles_sanitized_valid_b)
        prediction_sas_b = SAscore(mols_list_b)
        
        unique_smiles_b = list(np.unique(smiles_sanitized_valid_b))
        percentage_unq_b = (len(unique_smiles_b)/len(smiles_sanitized_valid_b))*100
        
        # plot both distributions together and compute the % of valid generated by the biased model 
        Utils.plot_hist_both(prediction_usp7_unb,prediction_usp7_b,prediction_sas_unb,prediction_sas_b,self.config.mols_to_generate,valid_unb,valid_b)
    
        
    def drawMols(self):
        """
        Function that draws the chemical structure of given list of compounds

        Parameters:
        -----------
        self: it contains the Generator and the configuration parameters

        Returns
        -------
        This function returns a figure with the specified number of molecules
        """
        
        DrawingOptions.atomLabelFontSize = 50
        DrawingOptions.dotsPerAngstrom = 100
        DrawingOptions.bondLineWidth = 3
        DrawingOptions.addStereoAnnotation = True  
        
        smiles_generated = ['CC(C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12',
                              'Cc1cccc(C)c1C(=O)NC1CCCNC1=O','CC1CCC(C)C12C(=O)Nc1ccccc12',
                              'NC(Cc1ccc(O)cc1)C(=O)O','CC(=O)OCC(=O)C1(OC(C)=O)CCC2C3CCC4=CC(=O)CCC4(C)C3CCC21C',
                              'CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1',
                              'CC1(C)CCC(C)(C)c2cc(C(=O)Nc3ccc(C(=O)O)cc3)ccc21','CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1',
                              'CCC(c1ccccc1)c1ccc(OCCN(CC)CC)cc1','Cc1cccc(C)c1NC(=O)CN(C)C(=O)C(C)C','CC(C)C(CO)Nc1ccnc2cc(Cl)ccc12',
                              'Cc1ccc(C(=O)Nc2ccc(C(C)C)cc2)cc1Nc1nccc(-c2ccc(C(F)(F)F)cc2)n1','CN(C)CCCNC(=O)CCC(=O)Nc1ccccc1',
                              'CC(C)CC(=O)N(Cc1ccccc1)C1CCN(Cc2ccccc2)CC1']
        
        known_drugs = ['C[C@@H]1[C@H]2C3=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@@]4(C)[C@]3(C)CC[C@@]2(C(=O)O)CC[C@H]1C',
                        'O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1','CCC1(c2ccc(N)cc2)CCC(=O)NC1=O',
                        'CC(N)(Cc1ccc(O)cc1)C(=O)O','CC(=O)O[C@]1(C(C)=O)CC[C@H]2[C@@H]3C=C(C)C4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]21C',
                        'COc1c(N2C[C@@H]3CCCN[C@@H]3C2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12',
                        'C=C(c1ccc(C(=O)O)cc1)c1cc2c(cc1C)C(C)(C)CCC2(C)C','O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O',
                        'CCN(CC)CCOc1ccc(Cc2ccccc2)cc1', 'CCN(CC)CC(=O)Nc1c(C)cccc1C', 'CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12', 
                        'Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1','O=C(CCCCCCC(=O)Nc1ccccc1)NO',
                        'CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1']
        
        legends = ['Ursolic acid', 'Thalidomide', 'Aminoglutethimide',
                    'Racemetyrosine', 'Megestrol acetate', 'Moxifloxacin',
                    'Bexarotene', 'Ciproflaxicin', 'Tesmilifene', 'Lidocaine', 
                    'Hydroxycloroquine', 'Nilotilib', 'Vorinostat', 'Fentanyl']
        
        generated_mols = Utils.smiles2mol(smiles_generated)
        drugs_mols = Utils.smiles2mol(known_drugs)
    
        img = Draw.MolsToGridImage(generated_mols, molsPerRow=3, subImgSize=(300,300))
        img.show()
        
        img = Draw.MolsToGridImage(drugs_mols, molsPerRow=3, subImgSize=(300,300),legends=legends)
        img.show()
        
    def select_best_stereoisomers(self):  

        
        # df = DataLoader().load_promising_mols(self.config)
        # # # sort predictions
        # df_sorted = df.sort_values('pic50',ascending = False)
        
        # smiles_sorted = df_sorted['smiles'].tolist()
        
        
        
        # smiles_sorted = ['CC(C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12',
                              # 'Cc1cccc(C)c1C(=O)NC1CCCNC1=O','CC1CCC(C)C12C(=O)Nc1ccccc12',
                              # 'NC(Cc1ccc(O)cc1)C(=O)O','CC(=O)OCC(=O)C1(OC(C)=O)CCC2C3CCC4=CC(=O)CCC4(C)C3CCC21C',
                              # 'CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1','CC1(C)CCC(C)(C)c2cc(C(=O)Nc3ccc(C(=O)O)cc3)ccc21','CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1',
                              # 'CCC(c1ccccc1)c1ccc(OCCN(CC)CC)cc1','Cc1cccc(C)c1NC(=O)CN(C)C(=O)C(C)C','CC(C)C(CO)Nc1ccnc2cc(Cl)ccc12',
                              # 'Cc1ccc(C(=O)Nc2ccc(C(C)C)cc2)cc1Nc1nccc(-c2ccc(C(F)(F)F)cc2)n1','CN(C)CCCNC(=O)CCC(=O)Nc1ccccc1',
                              # 'CC(C)CC(=O)N(Cc1ccccc1)C1CCN(Cc2ccccc2)CC1']
        
        smiles_sorted = ['CC(C)C(=O)Nc1ccc(C(=O)Nc2ccc(S(C)(=O)=O)cc2)cc1',
                         'Cc1cc(-c2ccccc2)c(C#N)c(=N)o1',
                         'CC(CCC(=O)NC(C)(C)CC(=O)n1cc(C(N)=O)c2ccc(Cl)cc2c1=O)c1ccccc1',
                         'C=CCC1CC(C)CC(C)(C)CCC12C(=O)C(C=C)C(C)(C)C2=O',
                         'CN(C)C(OCCOC(=O)c1ccccc1)C(=O)Cn1cnc2ccccc2c1=O',
                         'C=C(C)C1CCC2(C)CCC3(C)C(=CC(=O)C4C5(C)CCC(O)C(C)(C)C5CCC43C)C12'  ]        
        
        
        opts = StereoEnumerationOptions(tryEmbedding=True,unique=True,onlyStereoGroups = False)

        for idx,smi in enumerate(smiles_sorted):
            print(smi)
            mols_list = Utils.smiles2mol([smi])
            stereo_isomers = list(EnumerateStereoisomers(mols_list[0],options=opts))
            
                   
            smiles_augmented = [Chem.MolToSmiles(stereo_mol) for stereo_mol in stereo_isomers]

            # if len(smiles_augmented) <2:
            #     keep_mols = stereo_isomers
            #     keep_mols = Utils.smiles2mol(smiles_augmented)
            
            # else:
                
            #prediction usp7 affinity
            prediction_usp7 = self.predictor_usp7.predict(smiles_augmented)
            
            # sort and get the original indexes
            out_arr = np.argsort(np.reshape(prediction_usp7,-1))
            
            keep_indices = list(out_arr[-self.config.max_stereoisomers:])
            
            keep_smiles = [smiles_augmented[k_idx] for k_idx in keep_indices]
            
            keep_mols = Utils.smiles2mol(keep_smiles)
                
            if self.config.draw_mols == 'true':
                # DrawingOptions.addStereoAnnotation = True
                DrawingOptions.atomLabelFontSize = 50
                DrawingOptions.dotsPerAngstrom = 100
                DrawingOptions.bondLineWidth = 3
                                             

                
                legends = []
                for i in keep_indices:
                     legends.append('pIC50 for USP7: ' + str(prediction_usp7[i]))
           
                img1 = Draw.MolsToGridImage([mols_list[0]], molsPerRow=1, subImgSize=(300,300))
                img2 = Draw.MolsToGridImage(keep_mols, molsPerRow=3, subImgSize=(300,300))
                img1.show()
                
                img2.show()
                img1.save('generated\mols_canonical_best' + str(idx) + '.png')
                img2.save('generated\mols_stereoisomers_best' + str(idx) + '.png')
    
        
        
        
    def filter_promising_mols(self):
        df1,df2 = DataLoader().load_promising_mols(self.config)
        
        #  df1_filtered = df1[(df1['pic50']>7) & (df1['sas']<4.5) & (df1['logp']>0)
        #                  & (df1['logp']<5.5) & (df1['qed']>0.15) & (df1['tpsa']<140)
        #                  & (df1['hdonors']<7) & (df1['hacceptors']<15) & 
        #                  (df1['rotable_bonds']<15) & (df1['n_rings']<5)]
        
        df1_filtered = df1[(df1['pic50']>7)]
        df2_filtered = df2[(df2['pic50']>6.7)]
        df_all = df2_filtered+df1_filtered
        
         # # sort predictions
        df_all = df1_filtered.sort_values('pic50',ascending = False) 
        
        df_all = df_all[df_all.columns[0]]
        # determining the name of the file
        file_path = 'filtered_hits_new.csv'
          
        # saving the excel
        df_all.to_csv(file_path,index=False)
        
        
        
        
        
        
        
                
        # loaded_smiles = ['CC(C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12',
        #                  'CC(=O)OCC(=O)C1(OC(C)=O)CCC2C3CCC4=CC(=O)CCC4(C)C3CCC21C',
        #                   'CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1']
        
        # mols_list = Utils.smiles2mol(loaded_smiles)
        # # stereoisomers can differ in the biological affinity, synthetizability
        # # and reactivity
        
        # opts = StereoEnumerationOptions(tryEmbedding=True,unique=True,onlyStereoGroups = False)

        # stereo_isomers = [list(EnumerateStereoisomers(mol,options=opts)) for mol in mols_list]
        
        # filtered_stereo_isomers_all = {}
        
        # for idx,stereo_mols in enumerate(stereo_isomers):
               
        #     smiles_augmented = [Chem.MolToSmiles(mol) for mol in stereo_mols]

        #     if len(smiles_augmented) <= 2:
        #         keep_mols = stereo_mols
        #         filtered_stereo_isomers_all[loaded_smiles[idx]] = smiles_augmented
        #         keep_mols = Utils.smiles2mol(smiles_augmented)
            
        #     else:
                
        #         #prediction usp7 affinity
        #         prediction_usp7 = self.predictor_usp7.predict(smiles_augmented)
                
        #         # sort and get the original indexes
        #         out_arr = np.argsort(prediction_usp7)
                
        #         keep_indices = list(out_arr[-self.config.max_stereoisomers:])
                
        #         keep_smiles = [smiles_augmented[k_idx] for k_idx in keep_indices]
                
        #         keep_mols = Utils.smiles2mol(keep_smiles)
        #         filtered_stereo_isomers_all[loaded_smiles[idx]] = keep_smiles
                
        #     if self.config.draw_mols == 'true':
        #         # DrawingOptions.addStereoAnnotation = True
        #         DrawingOptions.atomLabelFontSize = 50
        #         DrawingOptions.dotsPerAngstrom = 100
        #         DrawingOptions.bondLineWidth = 3
                                             

                
        #         legends = []
        #         for i in keep_indices:
        #              legends.append('pIC50 for USP7: ' + str(prediction_usp7[i]))
           
        #         img1 = Draw.MolsToGridImage([mols_list[idx]], molsPerRow=1, subImgSize=(300,300))
        #         img2 = Draw.MolsToGridImage(keep_mols, molsPerRow=3, subImgSize=(300,300))
        #         img1.show()
                
        #         img2.show()
        #         img1.save('generated\mols_canonical_' + str(idx) + '.png')
        #         img2.save('generated\mols_stereoisomers_' + str(idx) + '.png')
        
     
    def samples_generation(self):
        
        """
        Function to generate, draw and save molecules 
        """
        
        # self.generator_biased = DataLoader().load_generator(self.config,'biased')
        
        # _,trajectory = self.generator_biased.generate(self.config.mols_to_generate)
        
        # smiles_biased = [smile[1:-1] for smile in trajectory]
        
        # sanitized,valid = Utils.canonical_smiles(smiles_biased, sanitize=True, throw_warning=False) # validar 
        
        # sanitized_valid_repeated = []
        
        # for smi in sanitized:
        #     if len(smi)>1:
        #         sanitized_valid_repeated.append(smi)

        # sanitized_valid = list(set(sanitized_valid_repeated))
        
        # percentage_unq = (len(sanitized_valid)/len(sanitized_valid_repeated))*100
        
        # vld = (valid/self.config.mols_to_generate)*100
        
        # print("\nValid: ", vld)
        # print("\nUnique: ", percentage_unq)
        
        
        
        
        sanitized_valid = ['O=C(NCCOCCO)c1n(Cc2ccc(C(=O)N3CCC(CNC4C(=O)NC(=O)C4)CC3)cc2)c(C)cc1C',
                           'O=C(NC(Cc1ccccc1)C)CC(CC(=O)Nc1c(OC)ccc(C(=O)NCc2ccc(C(=O)NN)cc2)c1)(C)C',
                           'Clc1cc(C(=S)C(=O)c2c(CCc3ccc(C(=O)NCC(=O)N4OCCC(CO)C4)cc3)cccc2)ccc1',
                           'FC(F)(F)c1ccc(-c2nc(Nc3c(C)ccc(C(=O)Nc4ccc(C(C)C)cc4)c3)ncc2)cc1',
                           'Fc1c(N2C(=O)NC(CNC(=O)C)C2)cc2N(C3CC3)C=C(C(=O)[O-])C(=O)c2c1',
                           'S([OH0])(Nc1ccc(C(=O)Nc2ccc(C(F)(F)F)cc2)cc1)([CH0](C)C)C']
            
        #prediction usp7 affinity
        prediction_usp7 = self.predictor_usp7.predict(sanitized_valid)
               
        # prediction SA score
        mols_list = Utils.smiles2mol(sanitized_valid)
        prediction_sas = SAscore(mols_list)

        Utils.plot_hist(prediction_sas,self.config.mols_to_generate,valid,"sas")
        Utils.plot_hist(prediction_usp7,self.config.mols_to_generate,valid,"usp7")
         
        
        # if self.config.mols_to_draw > 0:
        #     drawing = True
            
        #     DrawingOptions.atomLabelFontSize = 50
        #     DrawingOptions.dotsPerAngstrom = 100
        #     DrawingOptions.bondLineWidth = 3
            
        #     batch = 0
        #     while drawing:
        #         input_str = input("Press 'enter' to keep drawing mols: ")
                
        #         if len(input_str) == 0:
                        
    
        #             ind = np.random.randint(0, len(mols_list), self.config.mols_to_draw)
        #             mols_to_draw = [mols_list[i] for i in ind]
                    
        #             legends = []
        #             for i in ind:
        #                 legends.append('pIC50 for USP7: ' + str(round(prediction_usp7[i],2)))
                    
        #             img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=1, subImgSize=(300,300), legends=legends)
                        
        #             img.show()
        #             img.save('generated//mols_' + str(batch) +'.png')
                    
        #         else:
        #             drawing = False
        
        with open("generated//sample_mols_newpred_rl.smi", 'w') as f:
            f.write("Number of molecules: %s\n" % str(len(sanitized_valid)))
            f.write("Percentage of valid and unique molecules: %s\n\n" % str(vld))
            f.write("SMILES, pIC50, SAS, MW, logP, QED\n")
            for i,smi in enumerate(sanitized_valid):
                mol = mols_list[i]
                

                q = QED.qed(mol)
                mw, logP = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
                data = str(sanitized_valid[i]) + " ," +  str(np.round(prediction_usp7[i],2)) + " ," + str(np.round(prediction_sas[i],2)) + " ,"  + str(np.round(mw,2)) + " ," + str(np.round(logP,2)) + " ," + str(np.round(q,2))
                f.write("%s\n" % data)  
        
        
    