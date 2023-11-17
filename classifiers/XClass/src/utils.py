#!/usr/bin/env python3

# XClass Classifier - Utilities
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from config import StaticRepParams as SRP
from config import ClassDocRepParams as CDRP
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
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=False)
        parser.add_argument("--random_state", type=int, default=SRP.random_state)
        parser.add_argument("--lm_type", type=str, default=SRP.lm_type)
        parser.add_argument("--vocab_min_occurrence", type=int, default=SRP.vocab_min_occurrence)
        parser.add_argument("--layer", type=int, default=SRP.layer)
        parser.add_argument("--T", type=int, default=CDRP.T)
        parser.add_argument("--attention_mechanism", type=str, default=CDRP.attention_mechanism)
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

