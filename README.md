# Artificial Intelligence for Prediction of Biological Activities and Generation of molecular hits using Stereochemical Information

<p align="justify"> In this work, we develop a method for generating targeted hit compounds by applying deep reinforcement learning and attention mechanisms to predict binding affinity against a biological target while considering stereochemical information. We demonstrated the importance of attention mechanisms to capture long-range dependencies in molecular sequences. Due to the importance of stereochemical information for the binding mechanism, this information was employed both in the prediction and generation processes. To identify the most promising hits, we apply the self-adaptive multi-objective optimization strategy. Moreover, to ensure the existence of stereochemical information, we consider all the possible enumerated stereoisomers to provide the most appropriate 3D structures. We evaluated this approach against the Ubiquitin-Specific Protease 7 (USP7) by generating
putative inhibitors for this target. Our methodology identify the regions of
the generated molecules that are important for the interaction with the receptor’s
active site. Also, the obtained results demonstrate that it is possible to discover
synthesizable molecules with high biological affinity for the target, containing the
indication of their optimal stereochemical conformation. </p>


## Model Dynamics Architecture
<p align="center"><img src="/generated/Fig1.jpg" width="90%" height="90%"/></p>

## Data Availability
### Molecular Generator
- **train_chembl_22_clean_1576904_sorted_std_final:** Train set extracted from ChEMBL 22
- **test_chembl_22_clean_1576904_sorted_std_final:** Test set extracted from ChEMBL 22
### USP7 inhibitors
- **data_usp7.smi:** pIC50 against USP7 + augmented dataset with corresponding SMILES 
 

## Requirements:
- Python 3.8.12
- Tensorflow 2.3.0
- Numpy 
- Pandas
- Scikit-learn
- Itertools
- Matplotlib
- Seaborn
- Bunch
- tqdm
- rdkit 2021.03.4

## Usage 
Run main.py file to implement the described dynamics. Loads the pre-trained models and optimizes the Generator using reinforcement learning to obtain a set of putative inhibitors of the USP7 containing its stereochemical information. There is a .json configuration file to change the parameters as desired.

### Running the best configuration
```
python main.py 
```
