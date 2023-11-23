#!/usr/bin/env python3

# XClass Classifier Configurations
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# -----------------------------------------------------------------------------
# Constants
class Constants:
    DPATH_DATA = Path('../../datasets/SBIC/')
    DPATH_CACHED = Path('../XClass/_cached')
    DPATH_MODELS = Path('../XClass/_models')
    FPATH_DATA = Path('SBIC.v2.agg.cmb_processed.csv')
    FPATH_LOGS = Path('xclass_run.log')
    FPATH_STATICREPS_DATA = Path('stage01_staticreps_data.pickle')
    FPATH_STATICREPS_TOKENS = Path('stage01_staticreps_tokens.pickle')
    FPATH_CLASSDOCREPS_DATA = Path('stage02_classdocreps_data.pickle')
    FPATH_DOCCLASSALIGN_DATA = Path('stage03_docclassalign_data.pickle')
    FPATH_SELECTED_DOCIDS = Path('stage04_selected_docids.json')
    FPATH_SELECTED_LABELSTRUE = Path('stage04_selected_labelstrue.txt')
    FPATH_SELECTED_LABELSPRED = Path('stage04_selected_labelspred.txt')
    FPATH_SELECTED_TEXTS = Path('stage04_selected_texts.txt')
    FPATH_SELECTED_FINAL = Path('stage04_selected_final.csv')
    MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')


class StaticRepParams:
    random_state = 42
    lm_type = 'bbu'
    vocab_min_occurrence = 5
    layer = 12  # last layer of BERT


class ClassDocRepParams:
    random_state = StaticRepParams.random_state
    lm_type = StaticRepParams.lm_type
    layer = StaticRepParams.layer
    T = 100
    attention_mechanism = 'mixture'


class DocClassAlignParams:
    random_state = StaticRepParams.random_state
    pca = 64                # number of dimensions projected to in PCA; use -1 to skip PCA
    cluster_method = 'gmm'  # options are "gmm" or "kmeans"


class ClassifyPrepParams:
    confidence_threshold = 0.5


# class ClassifyTrainParams:
#     # data_dir = Constants.DPATH_DATA
#     model_name_or_path = 'bert-base-uncased'
#     task_name = '_cached'
#     output_dir = 'OUTPUT_DIR'
#     do_train = True
#     do_eval = True
#     evaluate_during_training = True
#     learning_rate = 5e-5
#     num_train_epochs = 3.0
#     max_seq_length = 512
#     per_gpu_train_batch_size = 16
#     per_gpu_eval_batch_size = 16
#     logging_steps = 100000
#     save_steps = -1 


class Hyperparams:
    train_mode = 'original'
    training_orig = {
        'trainfrac': 0.75, 
        'testfrac': 0.125, 
        'shuffle': True
    }
    training_misc = {
        'trainfrac': 0.75, 
        'shuffle': True
    }
    model_Bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            # "bert-base-multilingual-uncased", 
            num_labels = 6,                     # the total number of class labels
            output_attentions = False,          # should the model return attentions weights?
            output_hidden_states = False,       # should the model return all of the hidden states?
    )
    hyperparams = {    
        'batch_size': 16,  
        'epochs': 3, 
        'optimizer': AdamW(model_Bert.parameters(), lr=5e-5)
    }


# Data Imports
class Data:
    def __init__(self):
        self.fpath_data = Constants.DPATH_DATA/Constants.FPATH_DATA
        self.data = None
        self.init(self.fpath_data)

    def init(self, fpath_data):
        fpath = Path(fpath_data)
        self.fpath_data = fpath
        self.data = pd.read_csv(fpath).rename(
            columns={'post': 'text', 'label': 'label_id', 'target_minority': 'label_name'})

