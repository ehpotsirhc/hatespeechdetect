from pathlib import Path
import itertools
import operator
import os

import numpy as np
import pandas as pd

linewidth = 200
np.set_printoptions(linewidth=linewidth)
np.set_printoptions(precision=3, suppress=True)

from collections import Counter

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from transformers import BertModel, BertTokenizer

MODELS = {
    'bbc': (BertModel, BertTokenizer, 'bert-base-cased'),
    'bbu': (BertModel, BertTokenizer, 'bert-base-uncased')
}

# all paths can be either absolute or relative to this utils file
DATA_FOLDER_PATH = os.path.join('..', 'data', 'datasets')
INTERMEDIATE_DATA_FOLDER_PATH = os.path.join('..', 'data', 'intermediate_data')
# this is also defined in run_train_text_classifier.sh, make sure to change both when changing.
FINETUNE_MODEL_PATH = os.path.join('..', 'models')


def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


def dot_product_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b))


def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)


def pairwise_distances(x, y):
    return cdist(x, y, 'euclidean')


def most_common(L):
    c = Counter(L)
    return c.most_common(1)[0][0]


def error_analysis(labels_true, labels_pred, **kwargs):
    fpath_test_text = Path(DATA_FOLDER_PATH)/'SBIC/dataset.txt'
    labels = pd.read_csv(Path(DATA_FOLDER_PATH)/'SBIC/classes.txt', header=None)[0].to_list()
    test_text = pd.read_csv(fpath_test_text, header=None)[0].to_list()
    
    n_err = kwargs['n_err'] if 'n_err' in kwargs else 5   # default to 5
    indices_wrong = np.argwhere((labels_pred==labels_true) == False).flatten()
    n_correct = (np.array(labels_pred)==np.array(labels_true)).sum()
    n_total = len(labels_true)

    if indices_wrong.size >= n_err:
        idx_examples = np.random.choice(indices_wrong, size=n_err, replace=False).tolist()
        labels_pred = np.array(labels_pred).take(idx_examples).tolist()
        labels_true = np.array(labels_true).take(idx_examples).tolist()
        examples = np.array(test_text).take(idx_examples).tolist()
        combined = list(zip(idx_examples, labels_pred, labels_true, examples))
        return [{'index':e[0], 'label_pred':(labels[e[1]], e[1]), 'label_true':(labels[e[2]], e[2]), 'text':e[3]} for e in combined]
    else:
        print('There are less than %s incorrect examples.' % n_err)
        return []


def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
    confusion = confusion_matrix(true_class, predicted_class)
    if output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
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

        errors = error_analysis(true_class, predicted_class)
        print('\nExamples of Incorrect Predictions...')
        [print('- '*25, '\n', e) for e in errors]
        print()

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
