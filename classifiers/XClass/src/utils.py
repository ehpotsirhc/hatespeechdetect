#!/usr/bin/env python3

# XClass Classifier - Utilities
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from .config import Constants
from .config import StaticRepParams as SRP
from .config import ClassDocRepParams as CDRP
from .config import DocClassAlignParams as DCAP
from .config import ClassifyPrepParams as CPP
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import logging, sys, argparse, time, numpy as np, re, json, os

# =================================================================================================

# -----------------------------------------------------------------------------
# handles the main program's bootstrapping
class Bootstrap:
    @staticmethod
    def logging_init():
        logpath = Constants.DPATH_LOGS/Constants.FPATH_LOG_MAIN
        os.makedirs(logpath.parent) if not logpath.parent.exists() else None
        logging.basicConfig(
            filename=logpath, 
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
        parser.add_argument('--usecached', action=argparse.BooleanOptionalAction)
        parser.add_argument('--evalonly', action=argparse.BooleanOptionalAction)
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
    def load_classnames(df_data):
        classnames = df_data.groupby([df_data.label_name.str.lower(), df_data.label_id])\
            .size().reset_index().sort_values(by='label_id').iloc[:,0].tolist()
        return classnames
    
    @staticmethod
    def load_text(dfcol_texts):
        raw_txt = dfcol_texts.tolist()
        clean_txt = dfcol_texts.apply(TextProc.clean_str).tolist()
        TextProc.text_statistics(raw_txt, 'raw_txt')
        TextProc.text_statistics(clean_txt, 'cleaned_txt')
        return clean_txt
    
    @staticmethod
    def load_labels(dfcol_labelid):
        return dfcol_labelid.tolist()

    @staticmethod
    def write_json(data, fpath_output, *args):
        logging.info(args[0]) if len(args) > 0 else None
        fpath_output = Path(fpath_output)
        os.makedirs(fpath_output.parent) if not fpath_output.parent.exists() else None
        with open(fpath_output, 'w') as f:
            json.dump(data, f, indent=4)


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
# text processing utils
class TextProc:
    @staticmethod
    def clean_str(string):
        if type(string) is not str or len(string) <= 1 or string is None:
            return 'empty string'
        else:
            string = TextProc.clean_html(string)
            string = TextProc.clean_email(string)
            string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
            string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    @staticmethod
    def clean_html(string: str):
        left_mark = '&lt;'
        right_mark = '&gt;'
        # for every line find matching left_mark and nearest right_mark
        while True:
            next_left_start = string.find(left_mark)
            if next_left_start == -1:
                break
            next_right_start = string.find(right_mark, next_left_start)
            if next_right_start == -1:
                print("Right mark without Left: " + string)
                break
            # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
            # clean_html.clean_links.append(string[next_left_start: next_right_start + len(right_mark)])
            string = string[:next_left_start] + " " + string[next_right_start + len(right_mark):]
        return string

    @staticmethod
    def clean_email(string: str):
        return " ".join([s for s in string.split() if "@" not in s])

    @staticmethod
    def text_statistics(text, name="default"):
        sz = len(text)
        tmp_text = [s.split(" ") for s in text]
        tmp_list = [len(doc) for doc in tmp_text]
        len_max = max(tmp_list)
        len_avg = np.average(tmp_list)
        len_std = np.std(tmp_list)
        print(f"\n### Dataset statistics for {name}: ###")
        print('# of documents is: {}'.format(sz))
        print('Document max length: {} (words)'.format(len_max))
        print('Document average length: {} (words)'.format(len_avg))
        print('Document length std: {} (words)'.format(len_std))
        print(f"#######################################")


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

