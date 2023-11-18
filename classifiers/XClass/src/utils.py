#!/usr/bin/env python3

# XClass Classifier - Utilities
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from config import Constants
from config import StaticRepParams as SRP
from config import ClassDocRepParams as CDRP
from config import DocClassAlignParams as DCAP
from config import ClassifyPrepParams as CPP
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import logging, sys, argparse, time, numpy as np

# =================================================================================================

# -----------------------------------------------------------------------------
# handles the main program's bootstrapping
class Bootstrap:
    @staticmethod
    def logging_init():
        logging.basicConfig(
            filename='run.log', 
            filemode='w', 
            format='%(asctime)s - [%(levelname)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S', 
            level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        return logging

    @staticmethod
    def argparse_init():
        fpath_dataset = Constants.DPATH_DATA/Constants.FPATH_DATA
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default=fpath_dataset)
        parser.add_argument('--random_state', type=int, default=SRP.random_state)
        parser.add_argument('--lm_type', type=str, default=SRP.lm_type)
        parser.add_argument('--vocab_min_occurrence', type=int, default=SRP.vocab_min_occurrence)
        parser.add_argument('--layer', type=int, default=SRP.layer)
        parser.add_argument('--T', type=int, default=CDRP.T)
        parser.add_argument('--attention_mechanism', type=str, default=CDRP.attention_mechanism)
        parser.add_argument('--pca', type=int, default=DCAP.pca)
        parser.add_argument('--cluster_method', type=str, default=DCAP.cluster_method)
        parser.add_argument('--confidence_threshold', default=CPP.confidence_threshold)
        args = parser.parse_args()
        return args


# -----------------------------------------------------------------------------
# utilities for Static Representations
class StaticRepUtils:
    @staticmethod
    def tensor_to_numpy(tensor):
        return tensor.clone().detach().cpu().numpy()


# -----------------------------------------------------------------------------
# utilities for Class Oriented Document Representations
class ClassDocRepUtils:
    def cosine_similarity_embeddings(emb_a, emb_b):
        return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


# -----------------------------------------------------------------------------
# handles data-related basic tasks
class DataUtils:
    @staticmethod
    def load_classnames(dfcol_classes):
        return sorted(dfcol_classes.str.lower().unique())
    
    @staticmethod
    def load_text(dfcol_texts):
        return dfcol_texts.apply(lambda txt: txt.lower()).tolist()
    
    @staticmethod
    def load_labels(dfcol_labelid):
        return dfcol_labelid.tolist()


# -----------------------------------------------------------------------------
# handles the main program's timers
class Timer:
    start, finish, total = -1, -1, -1

    @staticmethod
    def start():
        Timer.start = time.time()
        return Timer.start

    @staticmethod
    def finish():
        Timer.finish = time.time()
        Timer.total = Timer.finish - Timer.start
        return Timer.finish


# -----------------------------------------------------------------------------
# functions for evaluating data
class EvalUtils:
    # def error_analysis(labels_true, labels_pred, **kwargs):
    #     fpath_test_text = kwargs['fpath_test_text'] if 'fpath_test_text' in kwargs else Path(DATA_FOLDER_PATH)/'SBIC/dataset.txt'
    #     labels = pd.read_csv(Path(DATA_FOLDER_PATH)/'SBIC/classes.txt', header=None)[0].to_list()
    #     test_text = pd.read_csv(fpath_test_text, header=None)[0].to_list()
        
    #     n_err = kwargs['n_err'] if 'n_err' in kwargs else 5   # default to 5
    #     indices_wrong = np.argwhere((labels_pred==labels_true) == False).flatten()
    #     n_correct = (np.array(labels_pred)==np.array(labels_true)).sum()
    #     n_total = len(labels_true)

    #     if indices_wrong.size >= n_err:
    #         idx_examples = np.random.choice(indices_wrong, size=n_err, replace=False).tolist()
    #         labels_pred = np.array(labels_pred).take(idx_examples).tolist()
    #         labels_true = np.array(labels_true).take(idx_examples).tolist()
    #         examples = np.array(test_text).take(idx_examples).tolist()
    #         combined = list(zip(idx_examples, labels_pred, labels_true, examples))
    #         return [{'index':e[0], 'label_pred':(labels[e[1]], e[1]), 'label_true':(labels[e[2]], e[2]), 'text':e[3]} for e in combined]
    #     else:
    #         print('There are less than %s incorrect examples.' % n_err)
    #         return []

    def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
        confusion = confusion_matrix(true_class, predicted_class)
        if output_to_console:
            print('%s Evaluating %s' % ('-'*20, '-'*20))
            print(confusion)
        n_correct = (np.array(true_class)==np.array(predicted_class)).sum()
        accuracy = accuracy_score(true_class, predicted_class)
        precision_macro = precision_score(true_class, predicted_class, average='macro')
        precision_micro = precision_score(true_class, predicted_class, average='micro')
        recall_macro = recall_score(true_class, predicted_class, average='macro')
        recall_micro = recall_score(true_class, predicted_class, average='micro')
        f1_macro = f1_score(true_class, predicted_class, average='macro')
        f1_micro = f1_score(true_class, predicted_class, average='micro')
        if output_to_console:
            print("accuracy: %s  (%s/%s)" % (accuracy, n_correct, len(true_class)))
            print("precision macro: %s" % precision_macro)
            print("precision micro: %s" % precision_micro)
            print("recall macro: %s" % recall_macro)
            print("recall micro: %s" % recall_micro)
            print("F1 macro: " + str(f1_macro))
            print("F1 micro: " + str(f1_micro))

            # errors = error_analysis(true_class, predicted_class)
            # print('\nExamples of Incorrect Predictions...')
            # [print('- '*25, '\n', e) for e in errors]
            # print()

        if return_tuple:
            return confusion, f1_macro, f1_micro, accuracy, precision_macro, precision_micro, recall_macro, recall_micro
            
        else:
            return {
                "confusion": confusion.tolist(),
                "f1_macro": f1_macro,
                "f1_micro": f1_micro, 
                "accuracy": accuracy, 
                "precision_macro": precision_macro, 
                "precision_micro": precision_micro, 
                "recall_macro": recall_macro, 
                "recall_micro": recall_micro
            }

