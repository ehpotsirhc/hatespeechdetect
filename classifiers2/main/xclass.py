#!/usr/bin/env python3

# XClass Classifier
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
import sys
sys.path.append('../')

from pathlib import Path
from XClass.src.config import Data, Constants
from XClass.src import utils, preproc, classify
from XClass.src.config import Hyperparams

from BERT.src.helpers2 import Helpers2 as BertHlp
from BERT.src.config import Constants as BertConst, Hyperparams as BertHparams, Data as BertData
from BERT.src.preproc import Preproc as BertPreproc
from BERT.src import classify as BertClassify
import torch, time

# =================================================================================================
# Main Function

def xclass_pretrain(args, texts, classnames, labels):
    StaticReps = preproc.StaticReps()
    ClassDocReps = preproc.ClassDocReps()
    DocClassAlign = preproc.DocClassAlign()
    ClassifyPrep = classify.Prep()
    StaticReps.main(args, texts)            # compute vocab from text
    ClassDocReps.main(args, classnames)     # associate vocab with classes; build doc representations
    DocClassAlign.main(args, classnames)    # align documents to their closest classes
    ClassifyPrep.main(args, texts, classnames, labels)  # filter for the high-confidence texts; preps data for final training


def bert_main(args, logger, model_main, df_training, **kwargs):
    hyparams = kwargs['hyparams'] if 'hyparams' in kwargs else {}
    fpath_modelcached = Constants.DPATH_MODELS/Constants.FPATH_MODEL
    
    BertHlp.seed_everything()
    BertConst.GPU_DEVICE = BertHlp.gpu_init(showmsg=True)[0]
    BertConst.MODEL = model_main
    
    logger.info('Running in full (train/validate/test) mode...')
    print('Unsaved models will be overwritten. Use Control-C to terminate immediately...\n')
    time.sleep(1)

    train_set, val_set, test_labels, test_goldlabels, test_texts = BertPreproc.training_split_and_tensorify(df_training)
    BertPreproc.tvs_stats(logger, train_set, val_set, test_texts)

    if not args.testing_only:
        torch.cuda.empty_cache()
        model_main.cuda()
        logger.info('Beginning language model training...')
        BertClassify.model_train(logger, model_main, train_set, val_set, **hyparams)

        logger.info('Caching trained language model to "%s"...' % str(fpath_modelcached))
        print('\n')
        BertClassify.model_save(BertConst.MODEL, fpath_modelcached)

    logger.info('Model Evaluation...')
    BertHlp.seed_everything()
    torch.cuda.empty_cache()
    model_main.load_state_dict(torch.load(fpath_modelcached))
    model_main.cuda()
    errors = BertClassify.model_error(logger, test_texts, test_goldlabels, hyparams=hyparams, n_err=5)
    print()
    logger.info('Examples of Incorrect Predictions...')
    for incorrect in errors:
        print('- '*25)
        logger.info(incorrect)
    print()


def main(args, logger):
    logger.info('Main function started'), logger.info(args)
    DataUtils = utils.DataUtils()

    if args.dataset and args.dataset.name.endswith('.csv'):
        Data.init(args.dataset)

    texts = DataUtils.load_text(Data.data.text)
    classnames = DataUtils.load_classnames(Data.data.label_name)
    labels = DataUtils.load_labels(Data.data.label_id)
    
    xclass_pretrain(args, texts, classnames, labels)
    
    print('\n  .....\n')
    logger.info('Beginning language model train/validate/test using the XClass-pretrained dataset...')
    BertHlp.seed_everything()
    BertDataObj = BertData()
    fpath_selected_final = Constants.DPATH_CACHED/Constants.FPATH_SELECTED_FINAL

    BertDataObj.init(fpath_selected_final)
    print(fpath_selected_final)
    bert_main(args, logger, Hyperparams.model_Bert, BertDataObj.training, hyparams=Hyperparams.hyperparams)


#  ----------------------------------------------------------------------------
# Run the Main Function
Data = Data()

if __name__ == '__main__':
    bootstrap, timer = utils.Bootstrap(), utils.Timer()
    argser, logger = bootstrap.argparse_init(), bootstrap.logging_init()
    timer.start()
    logger.info('Program started at %s' % timer.start)
    print()
    main(argser, logger)
    print()
    timer.finish()
    logger.info('Program finished at %s' % timer.finish)
    logger.info('Total Runtime: %.3fs  (%.0fm %.0fs)\n\n' % (timer.total, timer.total//60, timer.total%60))

