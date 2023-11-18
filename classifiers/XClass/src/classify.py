#!/usr/bin/env python3

# XClass Classifier - Classification Module
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from config import Constants
import logging, pickle
import numpy as np
from utils import EvalUtils

# =================================================================================================

# -----------------------------------------------------------------------------
# Prepare Data for Text Classifer Training
class Prep:
    @staticmethod
    def get_suffix(args):
        return 'pca%s.clus%s.%s-%s.%s-%s.%s' % (
            args.pca, args.cluster_method, args.lm_type, args.layer, \
                args.attention_mechanism, args.T, args.random_state)


    @staticmethod
    def read_aligned(suffix):
        with open(Constants.DPATH_CACHED/f'data.{suffix}.pickle', 'rb') as f:
            aligned = pickle.load(f)
        return aligned


    @staticmethod
    def get_doc_confidences(n_classes, n_docs, docs_to_class, distance):
        logging.info('Consolidating document confidences...')
        pseudo_doc_class_with_conf = [[] for _ in range(n_classes)]
        for i in range(n_docs):
            pseudo_doc_class_with_conf[docs_to_class[i]]\
                .append((distance[i][docs_to_class[i]], i))
        return pseudo_doc_class_with_conf
    

    @staticmethod
    def filter_for_confident(args, n_classes, pseudo_doc_class_with_conf, texts, docs_to_class):
        docs_selected = []
        for i in range(n_classes):
            pseudo_doc_class_with_conf[i] = sorted(pseudo_doc_class_with_conf[i])
            num_docs_to_take = int(len(pseudo_doc_class_with_conf[i]) * args.confidence_threshold)
            confident_documents = pseudo_doc_class_with_conf[i][:num_docs_to_take]
            confident_documents = [x[1] for x in confident_documents]
            docs_selected.extend(confident_documents)
        docs_selected = sorted(docs_selected)
        texts_selected = [texts[i] for i in docs_selected]
        classes_selected = [docs_to_class[i] for i in docs_selected]
        return (docs_selected, texts_selected, classes_selected)
    
    
    # -------------------------------------------------------------------------
    def main(self, args, texts, labels):
        logging.info('Preparing Data for Training...')
        suffix = self.get_suffix(args)
        data_aligned = self.read_aligned(suffix)

        docs_to_class = data_aligned['documents_to_class']
        distance = data_aligned['distance']
        n_docs, n_classes = distance.shape

        pseudo_doc_class_with_conf = self.get_doc_confidences(
            n_classes, n_docs, docs_to_class, distance)

        labels_true = labels

        docs_selected, texts_selected, classes_pred = self.filter_for_confident(
            args, n_classes, pseudo_doc_class_with_conf, texts, docs_to_class)

        classes_true = [labels_true[i] for i in docs_selected]

        np_classes_true, np_classes_pred = np.array(classes_true), np.array(classes_pred)


        EvalUtils.evaluate_predictions(classes_true, classes_pred)

