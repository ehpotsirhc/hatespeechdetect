#!/usr/bin/env python3

# BERT Classifier Preprocessing Functions
# Christophe Leung
# October 30, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from .helpers import tokenize_and_format
from .helpers2 import Helpers2 as hlp
from .config import Constants, Hyperparams
import pandas as pd, numpy as np, torch, os

# -----------------------------------------------------------------------------
# Data Preprocessing - Tokenize / Vectorize / Shuffle
class Preproc:
    @staticmethod
    def labels2tensors(arr_labels: np.ndarray, **kwargs):
        n_classes = kwargs['n_classes'] if 'n_classes' in kwargs else None
        arr_labels, vec_labels = arr_labels, []
        for l in arr_labels:
            vec_label = np.zeros(n_classes) if n_classes else np.zeros(len(set(arr_labels)))
            vec_label[l] = 1
            vec_labels.append(vec_label)
        return torch.tensor(np.array(vec_labels))


    @staticmethod
    def texts2tensors(arr_texts: np.ndarray):
        input_ids, attention_masks = tokenize_and_format(arr_texts)
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return (input_ids, attention_masks)


    @staticmethod
    def df2tensors(arr_labels, arr_texts, **kwargs):
        n_classes = kwargs['n_classes'] if 'n_classes' in kwargs else None
        labels_tensors = Preproc.labels2tensors(arr_labels, n_classes=n_classes)
        input_ids, attention_masks = Preproc.texts2tensors(arr_texts)
        return list(zip(input_ids, attention_masks, labels_tensors))


    @staticmethod
    def texts2set(arr_texts):
        test_input_ids, test_attention_masks = Preproc.texts2tensors(arr_texts)
        test_set = list(zip(test_input_ids, test_attention_masks))
        return test_set


    @staticmethod
    def dfshuffle(df: pd.DataFrame):
        hlp.seed_everything()
        return df.sample(frac=1).reset_index(drop=True)


    @staticmethod
    def train_val_split(df_loaded, **kwargs):
        trainfrac = float(kwargs['trainfrac']) if 'trainfrac' in kwargs else 0.8
        testfrac = float(kwargs['testfrac']) if 'testfrac' in kwargs else 0.0
        shuffle = bool(kwargs['shuffle']) if 'shuffle' in kwargs else False

        total = len(df_loaded)
        n_train, n_test = int(total * trainfrac), int(total * testfrac)
        n_val = total - n_train - n_test

        if shuffle == True:
            df_loaded = Preproc.dfshuffle(df_loaded)

        df_train = df_loaded[:n_train]
        df_val = df_loaded[n_train:(n_train+n_val)]
        df_test = df_loaded[(n_train+n_val):]

        return (df_train, df_val, df_test)


    @staticmethod
    def training_split_and_tensorify(df_training, **kwargs):
        mode = kwargs['mode'] if 'mode' in kwargs else Hyperparams.train_mode
        os.makedirs(Constants.DPATH_CACHED) if not Constants.DPATH_CACHED.exists() else None

        if mode=='original':
            df_train, df_val, df_test = Preproc.train_val_split(df_training, **Hyperparams.training_orig)
            df_train.to_csv(Constants.DPATH_CACHED/'vset_orig_train.tsv', sep='\t', index=False)
            df_val.to_csv(Constants.DPATH_CACHED/'vset_orig_val.tsv', sep='\t', index=False)
            df_test.to_csv(Constants.DPATH_CACHED/'vset_orig_test.tsv', sep='\t', index=False)
        
        elif mode=='dataset_as_test':
            # use the original training data with a modified test set; must run the "original" as least once first
            df_train_original = pd.read_csv(Constants.DPATH_CACHED/'vset_orig_train.tsv', sep='\t')
            df_val_original = pd.read_csv(Constants.DPATH_CACHED/'vset_orig_val.tsv', sep='\t')
            df_original = pd.concat([df_train_original, df_val_original], axis=0)
            df_train, df_val, df_test = Preproc.train_val_split(df_original, **Hyperparams.training_misc)
            
            # set/override df_test, the test set, to be the custom test set
            df_test = df_training
        
        else:
            df_train, df_val, df_test = Preproc.train_val_split(df_training, **Hyperparams.training_misc)
            df_test = pd.read_csv(Constants.DPATH_CACHED/'vset_orig_test.tsv', sep='\t')
            
        # convert dataframe data to tensors
        train_labels, train_texts = df_train.label_ID.values, df_train.sentence.values
        val_labels, val_texts = df_val.label_ID.values, df_val.sentence.values
        test_labels, test_texts = df_test.label_ID.values, df_test.sentence.values
        test_goldlabels = df_test.label_gold.values if 'label_gold' in df_test.columns else []
        n_classes = Hyperparams.model_Bert.classifier.out_features
        train_set = Preproc.df2tensors(train_labels, train_texts, n_classes=n_classes)
        val_set = Preproc.df2tensors(val_labels, val_texts, n_classes=n_classes)
        
        return (train_set, val_set, test_labels, test_goldlabels, test_texts)


    @staticmethod
    def tvs_stats(logger, train_set, val_set, test_text):
        trn, val, tst = len(train_set), len(val_set), len(test_text)
        logger.info('Dataset Stats (sizes)... train_set: %s, val_set: %s, test_text: %s, total: %s' % \
            (trn, val, tst, trn+val+tst))
        hlp.separator(msg='Training Dataset Stats')
        print('  train_set:', trn)
        print('    val_set:', val)
        print('  test_text:', tst)
        print('      TOTAL:', trn+val+tst)
        print()

