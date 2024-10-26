# Presentation

Implementation of a Natural Language Inference task based on the Stanford Natural Language Inference dataset : https://nlp.stanford.edu/projects/snli/

# Overview

We finetuned a pretrained Albert model (with on top a linear layer) on the SNLI corpus, in order to achieve the NLI task. 
We used the CrossEntropy loss (since it is a classification task), optimized with Adam and used accuracy as main evaluation metric.  

The scripts are the following : 
    - `training.py` : allows to train the model on the dataset (see below for usage)
    - `inference.ipynb` : allows to play with the trained model with customed sentences

# Setup of the environment
You first need to install the virtual environment with the following command:

`python -m venv venv`

Then you can activate it with the command:

`source venv/bin/activate`

You can now install the required libraries with

`pip install -r requirement.txt`

# Training 

Change the "BASE_RES_PATH" parameter in config_sample.yaml to your current directory (use pwd command to print it in the terminal).

Finally, you can run the code with : 

`python -m training config_sample.yaml`


