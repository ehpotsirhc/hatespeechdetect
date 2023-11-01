#!/usr/bin/env python3

# BERT Classifier Configurations
# Christophe Leung
# October 30, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from torch.optim import AdamW
from transformers import BertForSequenceClassification
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
class Constants:
    DPATH_DATA = Path('../data/')
    DPATH_MODEL = Path('../models/')
    # LABEL2ID = pd.read_json('../data/classes.json', orient='index').to_dict()[0]
    LABELS = pd.read_csv(DPATH_DATA/'classes.txt', header=None)[0].to_list()
    GPU_DEVICE = None
    MODEL = None


# Model Hyperparameters
class Hyperparams:
    train_mode = 'original'
    training_orig = {
        'trainfrac': 0.8, 
        'testfrac': 0.1, 
        'shuffle': True
    }
    training_misc = {
        'trainfrac': 0.8, 
        'shuffle': True
    }
    model_Bert = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-uncased", 
            num_labels = len(Constants.LABELS), # the number of class labels
            output_attentions = False,          # should the model return attentions weights?
            output_hidden_states = False,       # should the model return all of the hidden states?
    )
    hyperparams = {    
        'batch_size': 50,  
        'epochs': 10, 
        'optimizer': AdamW(model_Bert.parameters(), lr=1e-5, betas=(0.9, 0.555), eps=1e-8)
    }


# Data Imports
class Data:
    def __init__(self):
        self.fpath_training = Constants.DPATH_DATA/'SBIC.v2.agg.cmb_processed.csv'
        self.training = None
        self.init(self.fpath_training)

    def init(self, fpath_training):
        fpath = Path(fpath_training)
        self.fpath_training = fpath
        self.training = pd.read_csv(fpath).rename(columns={'post': 'sentence', 'label': 'label_ID'})

