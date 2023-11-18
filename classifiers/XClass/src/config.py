#!/usr/bin/env python3

# XClass Classifier Configurations
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
from transformers import BertModel, BertTokenizer

# -----------------------------------------------------------------------------
# Constants
class Constants:
    DPATH_DATA = Path('../../../datasets/SBIC/')
    DPATH_CACHED = Path('_cached')
    DPATH_MODELS = Path('_models')
    FPATH_DATA = Path('SBIC.v2.agg.cmb_processed.csv')
    # LABELS = pd.read_csv(Path(DPATH_DATA/'SBIC.v2.agg.cmb_classes.txt'), header=None)[0].to_list()
    # GPU_DEVICE = None
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


# Model Hyperparameters
class Hyperparams:
    train_mode = 'original'


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

