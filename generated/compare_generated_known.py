# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 09:28:28 2021

@author: tiago
"""
import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import QED
# from utils.sascorer_calculator import SAscore
# import pylev
from rdkit.Chem import rdFMCS

def load_generated_mols(filepath):

    if '.smi' in filepath:
        idx_smiles = 0
        
        raw_smiles = []
        
        with open(filepath, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)

            for idx,row in enumerate(it):
                if idx > 3:
                    try:
                        raw_smiles.append(row[idx_smiles])
                    except:
                        pass
        
       
        return list(set(raw_smiles))
    
    elif '.pkl' in filepath:
        df = pd.read_pickle(filepath)
        
        mols_all = list(df['molecules'])
        mols_d = list(df.loc[df["dominated"] == True, "molecules"])
        mols_nd = list(df.loc[df["dominated"] == False, "molecules"])
        return mols_all
        

def load_known_mols():
    # filepaths = ["cancer_drugs.csv","usp7_chembl.csv"]
    filepaths = ["inhibitors_Rita.smi"]
    raw_smiles = []
    name_mols = []
    for fp in filepaths:
        
        with open(fp, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)
            next(it, None)  # skip first item.    
            for ii,row in enumerate(it): 
                print(ii)     
                # if ii == 186:
                #     print('aqui')
                if fp == 'usp7_chembl.csv' :
                    
                    if len(row) == 2:
                        row_splited_0 = row[0].split(';')
                        row_splited_1 = row[1].split(';')
                        idx_smiles = 1
                        idx_pic50 = 6
                        idx_name = 0
                        smiles = row_splited_1[idx_smiles][1:-1]
                        name_mol = row_splited_0[idx_name]
                        pic50 = row_splited_1[idx_pic50][1:-1]
                    elif len(row) == 3:
                        row_splited_0 = row[0].split(';')
                        row_splited_2 = row[2].split(';')
                        idx_smiles = 1
                        idx_pic50 = 6
                        idx_name = 0
                        smiles = row_splited_2[idx_smiles][1:-1]
                        name_mol = row_splited_0[idx_name]
                        pic50 = row_splited_2[idx_pic50][1:-1]
                    else:
                        row_splited = row[0].split(';')
                        idx_smiles = 7
                        idx_pic50 = 12
                        idx_name = 0
                        smiles = row_splited[idx_smiles][1:-1]
                        name_mol = row_splited[idx_name]
                        pic50 = row_splited[idx_pic50][1:-1]

                    
                    try:
      
                        mol = Chem.MolFromSmiles(smiles, sanitize=True)
                        s = Chem.MolToSmiles(mol)

                        
                        if s not in raw_smiles and len(s)>10:
                            
                            if float(pic50) > 5:
                                raw_smiles.append(s)
                                name_mols.append(name_mol)
             
                    except:
                        print(smiles)
                elif fp == 'inhibitors_Rita.smi' : 

                    
                    idx_smiles = 0
                    idx_name = 1
                    
                    try:
                        
                        sm = row[idx_smiles]
                        # print(sm,row_splited[idx_pic50][1:-1])
                        mol = Chem.MolFromSmiles(sm, sanitize=True)
                        s = Chem.MolToSmiles(mol)
                        name_mol = row[idx_name]
                        if s not in raw_smiles and len(s)>10:
                            
                            raw_smiles.append(s)
                            name_mols.append(name_mol)
                            
             
                    except:
                        print(sm)
                    
                else:
                    row_splited = row[0].split(';')
                    
                    idx_smiles = 30
                    idx_name = 1
                    
                    try:
                        
                        sm = row_splited[idx_smiles][1:-1]
                        # print(sm,row_splited[idx_pic50][1:-1])
                        mol = Chem.MolFromSmiles(sm, sanitize=True)
                        s = Chem.MolToSmiles(mol)
                        name_mol = row_splited[idx_name]
                        if s not in raw_smiles and len(s)>10:
                            
                            raw_smiles.append(s)
                            name_mols.append(name_mol)
                            
             
                    except:
                        print(sm)

    return raw_smiles,name_mols
            
       
def find_similarities(generated,leads,similarity_measure,drug_names):
    """Loads and combines the required models into the dynamics of generating
    novel molecules"""
    similarities = pd.DataFrame()
    similarity = []
    most_similar_mol = []
    name_most_similar_mol = []
    for m in generated:    	
        
        if similarity_measure == 'Tanimoto_s':
            try:
                mol = Chem.MolFromSmiles(m, sanitize=True)
            	
                fp_m = AllChem.GetMorganFingerprint(mol, 3)   
    
                [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(n, sanitize=True), 3) for n in leads]
    
                dists = [DataStructs.TanimotoSimilarity(fp_m, AllChem.GetMorganFingerprint(Chem.MolFromSmiles(n, sanitize=True), 3)) for n in leads]    
                
                most_similar_mol.append(leads[dists.index(max(dists))])
                name_most_similar_mol.append(drug_names[dists.index(max(dists))])
                similarity.append(max(dists))
                
                
            except:
                similarity = ['nan']
                most_similar_mol = ['nan']
                print('Invalid: ' + m)
                
        elif similarity_measure == 'Tanimoto_mcs':
            
         
            ref_mol = Chem.MolFromSmiles(m, sanitize=True)
            numAtomsRefCpd = float(ref_mol.GetNumAtoms())
            dists= []
            for l in leads:
                
                try: 
                    target_mol = Chem.MolFromSmiles(l, sanitize=True)
                    numAtomsTargetCpd = float(target_mol.GetNumAtoms())
    
                    # if numAtomsRefCpd < numAtomsTargetCpd:
                    #     leastNumAtms = int(numAtomsRefCpd)
                    # else:
                    #     leastNumAtms = int(numAtomsTargetCpd)
            
                    pair_of_molecules = [ref_mol, target_mol]
                    numCommonAtoms = rdFMCS.FindMCS(pair_of_molecules, 
                                                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                                                    bondCompare=rdFMCS.BondCompare.CompareOrderExact, matchValences=True).numAtoms
                    dists.append(numCommonAtoms/((numAtomsTargetCpd+numAtomsRefCpd)-numCommonAtoms))
                except:
                    dists.append(-1)
                    print('Invalid: ' + l)
            
            # try:        
            most_similar_mol.append(leads[dists.index(max(dists))])
            name_most_similar_mol.append(drug_names[dists.index(max(dists))])
            similarity.append(max(dists))
            [dists.index(max(dists))]
            # except:
            #     print()
    
        # else:
        #     dists = [pylev.levenshtein(m, l) for l in leads]
        #     most_similar_mol.append(leads[dists.index(min(dists))])
        #     similarity.append(min(dists))
            
            

    similarities['generated'] = generated
    similarities['most similar drug'] = most_similar_mol
    similarities['Tanimoto similarity'] = similarity
    similarities['Name drug'] = name_most_similar_mol
    
    similarities_sorted = similarities.sort_values(by = 'Tanimoto similarity',ascending = False)
    
    return similarities_sorted
    
def draw_mols(smiles_list,title, names=False):

    DrawingOptions.atomLabelFontSize = 50
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3
    
    mols_list = [ Chem.MolFromSmiles(m, sanitize=True) for m in  smiles_list]
    
    if names!= False:
        legend_mols = []
        for i in range(0,len(mols_list)):
            legend_mols.append('Id: '+ str(names[i]))
    
        img = Draw.MolsToGridImage(mols_list, molsPerRow=4, legends=legend_mols, subImgSize=(300,300))
            
    else:
        img = Draw.MolsToGridImage(mols_list, molsPerRow=4, subImgSize=(300,300))
        
    img.show()
    img.save('mols_' + title + '.png')

if __name__ == '__main__':
    
    similarity_measure = 'Tanimoto_s' # Tanimoto_s, Tanimoto_mcs
    
    # Select the input file and the similarity measure 
    file_generated_mols = "pareto_generation_mols.pkl" #sample_mols_oldpred_rl.smi, sample_mols_newpred_rl.smi #generated_lead_comparison.smi or pareto_generation_mols.pkl 

    # Load the set of generated molecules
    generated_mols = load_generated_mols(file_generated_mols)
    
    # Load the set of molecules that interact with USP7 
    known_drugs,name_drugs = load_known_mols()
    
    # Compute the similarity 
    similarities = find_similarities(generated_mols,known_drugs,similarity_measure,name_drugs)
    
    # Select a subset of molecules
    # similarities_filtered = similarities_sorted[similarities_sorted["Tanimoto similarity"] > 0.43]
    similarities_filtered = similarities.head(25)
    
    draw_mols(similarities_filtered['generated'],'generated')
    draw_mols(similarities_filtered['most similar drug'],'known drugs',list(similarities_filtered['Name drug']))
    
  
    #All
    # # Tanimoto_S
    # generated = [3,4,5,8,15,18]
    # generated = ['CCC(c1ccccc1)c1ccc(OCCN(CC)CC)cc1','Cc1cccc(C)c1NC(=O)CN(C)C(=O)C(C)C','CC(C)C(CO)Nc1ccnc2cc(Cl)ccc12','Cc1ccc(C(=O)Nc2ccc(C(C)C)cc2)cc1Nc1nccc(-c2ccc(C(F)(F)F)cc2)n1','CN(C)CCCNC(=O)CCC(=O)Nc1ccccc1','CC(C)CC(=O)N(Cc1ccccc1)C1CCN(Cc2ccccc2)CC1']
    # known = [3(tesmilifene), 4(lidocaine), 5(hydroxycloroquine),8(nilotilib),15(vorinostat),18(fentanyl)]
    # known = ['CCN(CC)CCOc1ccc(Cc2ccccc2)cc1', 'CCN(CC)CC(=O)Nc1c(C)cccc1C', 'CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12', 'Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1','O=C(CCCCCCC(=O)Nc1ccccc1)NO','CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1']
    
    # # Tanimoto_mcs
    # generated = [1,7,11]
    # generated = ['CC(C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12','Cc1cccc(C)c1C(=O)NC1CCCNC1=O','CC1CCC(C)C12C(=O)Nc1ccccc12']
    # known = [1(ursolic acid), 7(thalidomide), 11(Aminoglutethimide)]
    # known = ['C[C@@H]1[C@H]2C3=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@@]4(C)[C@]3(C)CC[C@@]2(C(=O)O)CC[C@H]1C','O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1','CCC1(c2ccc(N)cc2)CCC(=O)NC1=O']
    
    # Best sample
    # # Tanimoto_S
    # generated = [6,7]
    # generated = ['CC1(C)CCC(C)(C)c2cc(C(=O)Nc3ccc(C(=O)O)cc3)ccc21','CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1']
    # known = [6(bexarotene), 7(ciproflaxicin)]
    

    # # Tanimoto_mcs
    # generated = [1,4,11]
    # generated = ['NC(Cc1ccc(O)cc1)C(=O)O','CC(=O)OCC(=O)C1(OC(C)=O)CCC2C3CCC4=CC(=O)CCC4(C)C3CCC21C','CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1']
    # known = [1(racemetyrosine), 4(megestrol acetate), 11(moxifloxacin)]
    # known = ['CC(N)(Cc1ccc(O)cc1)C(=O)O','CC(=O)O[C@]1(C(C)=O)CC[C@H]2[C@@H]3C=C(C)C4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]21C','COc1c(N2C[C@@H]3CCCN[C@@H]3C2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12']
    
    
    # rita_mols: old predictor
    # # Tanimoto_S
    # generated = [1,4,7,14,16,20]
    # generated = ['CC(C)Cc1ccc(C(=O)NS(C)(=O)=O)cc1','O=C(Nc1ccccc1)Nc1ccc(Cl)c(C(F)(F)F)c1','CC(C)Oc1ccc(NS(=O)(=O)c2ccc(O)cc2)cc1','CC(C)[SH](C)(=O)Nc1ccc(C(=O)Nc2ccc(C(F)(F)F)cc2)cc1','Cc1cccc(C(=O)Nc2ccc3c(c2)C(C)(C)CCC3(C)C)c1','CC(C)c1nc(C(=O)Nc2ccccc2)c(O)c(C(=O)c2ccc(F)cc2)c1-c1ccc(C(=O)O)cc1']
    # known = [1(reparixin),4(SORAFENIB),7(abt-751),14(leflunomide),16(bexarotene),20(atorvastatin)]
    # known = ['CC(C)Cc1ccc([C@@H](C)C(=O)NS(C)(=O)=O)cc1','CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1','COc1ccc(S(=O)(=O)Nc2cccnc2Nc2ccc(O)cc2)cc1','Cc1oncc1C(=O)Nc1ccc(C(F)(F)F)cc1','C=C(c1ccc(C(=O)O)cc1)c1cc2c(cc1C)C(C)(C)CCC2(C)C','CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O']
    
    # rita_mols: new predictor
    # # Tanimoto_S
    # generated = [2,3,19]
    # generated = ['CC(=O)c1ccc(C(C)=C(c2ccccc2)c2ccccc2)cc1','COc1ccc(CC(=O)Nc2ccc(C(=O)O)c(C)c2)cc1','CC(C)Nc1nc(N)c2ccccc2n1']
    # known = [2(tamoxifen),3(efaproxiral),19(imiquimod)]
    # known = ['CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1','Cc1cc(C)cc(NC(=O)Cc2ccc(OC(C)(C)C(=O)O)cc2)c1','CC(C)Cn1cnc2c(N)nc3ccccc3c21']