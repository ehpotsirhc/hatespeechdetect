#!/usr/bin/env python3

# XClass Classifier - Classification Module
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from config import Constants
import logging, pickle
import numpy as np, pandas as pd, os
from utils import EvalUtils, DataUtils

# =================================================================================================

# -----------------------------------------------------------------------------
# Prepare Data for Text Classifer Training
#   - filters for the high-confidence data from the pre-classified (aligned) dataset
#   - consolidates the required data necessary for the final training/classification
class Prep:
    @staticmethod
    def read_aligned():
        with open(Constants.DPATH_CACHED/Constants.FPATH_DOCCLASSALIGN_DATA, 'rb') as f:
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
        labelspred_selected = [docs_to_class[i] for i in docs_selected]
        return (docs_selected, texts_selected, labelspred_selected)


    @staticmethod
    def write_data(docs_selected, texts_selected, labelstrue_selected, labelspred_selected, classnames):
        fpath_selected_docids = Constants.DPATH_CACHED/Constants.FPATH_SELECTED_DOCIDS
        DataUtils.write_json(docs_selected, fpath_selected_docids, 
            'Caching confidence-filtered document IDs to "%s"' % fpath_selected_docids)
        
        os.makedirs(Constants.DPATH_CACHED) if not Constants.DPATH_CACHED.exists() else None
        
        assert len(texts_selected) == len(labelstrue_selected)
        
        fpath_selected_texts = Constants.DPATH_CACHED/Constants.FPATH_SELECTED_TEXTS
        # with open(fpath_selected_texts, 'w') as f:
        #     logging.info('Caching confidence-filtered texts to "%s"' % fpath_selected_texts)
        #     f.writelines(['%s\n'%line for line in texts_selected])
        
        # fpath_selected_labelstrue = Constants.DPATH_CACHED/Constants.FPATH_SELECTED_LABELSTRUE
        # with open(fpath_selected_labelstrue, 'w') as f:
        #     logging.info('Caching confidence-filtered true labels to "%s"' % fpath_selected_labelstrue)
        #     f.writelines(['%s\n'%line for line in labelstrue_selected])
        
        # fpath_selected_labelspred = Constants.DPATH_CACHED/Constants.FPATH_SELECTED_LABELSPRED
        # with open(fpath_selected_labelspred, 'w') as f:
        #     logging.info('Caching confidence-filtered predicted labels to "%s"' % fpath_selected_labelspred)
        #     f.writelines(['%s\n'%line for line in labelspred_selected])

        fpath_selected_final = Constants.DPATH_CACHED/Constants.FPATH_SELECTED_FINAL
        pd_selected_final = pd.DataFrame()
        pd_selected_final['post'] = texts_selected
        pd_selected_final['label_name'] = [classnames[label_id] for label_id in labelspred_selected]
        pd_selected_final['label_id'] = labelspred_selected
        pd_selected_final.to_csv(fpath_selected_final, index=None)
        


    # -------------------------------------------------------------------------
    def main(self, args, texts, classnames, labels_true):
        logging.info('Consolidating high-confidence aligned data for training...')
        data_aligned = self.read_aligned()

        docs_to_class = data_aligned['documents_to_class']
        distance = data_aligned['distance']
        n_docs, n_classes = distance.shape

        pseudo_doc_class_with_conf = self.get_doc_confidences(
            n_classes, n_docs, docs_to_class, distance)

        docs_selected, texts_selected, labelspred_selected = self.filter_for_confident(
            args, n_classes, pseudo_doc_class_with_conf, texts, docs_to_class)

        labelstrue_selected = [labels_true[doc_id] for doc_id in docs_selected]

        EvalUtils.evaluate_predictions(labelstrue_selected, labelspred_selected)

        self.write_data(docs_selected, texts_selected, labelstrue_selected, labelspred_selected, classnames)
        
