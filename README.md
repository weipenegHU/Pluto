# Pluto
MHC-I epitope presentation prediction based on transfer learning

Software install:

Download the zip file and uncompress it, you are good to go. However, make sure the following dependencies are installed before you use Pluto:
1. pandas 
2. numpy
3. tensorflow

Here are some illustration about how to used the scripts contained in this repository:  

__pretrain.py__ is for pretraining a model to recognize common features of the epitopes, which will be used to train the pluto model.  
Usage:  
`python pretrain.py [training dataset] [development dataset]`  
Example:  
`python pretrain.py data/pretrain/train_peptide.csv data/pretrain/dev_peptide.csv`  

__pluto.py__ is for training the final allele specific epitope presentation prediction model.  
Usage:  
`python pluto.py [HLA] [training dataset] [developent dataset]`  
Example:  
`python pluto.py A0101 data/A0101/train_peptide.csv data/A0101/dev_peptide.csv`  

__predict.py__ is for predicting epitope presentation for specific allele by using the already trained model.  
Usage:  
`python predict.py [checkpoint dir] [input file] [output file]`  
Example:  
if you want to predict a list of peptide will be presented by HLA-A0101, you can use:  
`python predict.py model/A0101/pretrain/ data/A0101/dev_peptide.csv output.csv`  
