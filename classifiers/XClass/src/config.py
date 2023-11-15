#!/usr/bin/env python3

# XClass Classifier Configurations
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
class Constants:
    DPATH_DATA = Path('../../../datasets/SBIC/')
    # DPATH_MODEL = Path('../models/')
    FPATH_DATA = Path('SBIC.v2.agg.cmb_processed.csv')
    # LABELS = pd.read_csv(Path(DPATH_DATA/'SBIC.v2.agg.cmb_classes.txt'), header=None)[0].to_list()
    # GPU_DEVICE = None
    # MODEL = None


# Model Hyperparameters
class Hyperparams:
    train_mode = 'original'


# Data Imports
class Data:
    def __init__(self):
        self.fpath_training = Constants.DPATH_DATA/Constants.FPATH_DATA
        self.training = None
        self.init(self.fpath_training)

    def init(self, fpath_training):
        fpath = Path(fpath_training)
        self.fpath_training = fpath
        self.training = pd.read_csv(fpath).rename(columns={'post': 'sentence', 'label': 'label_ID'})

