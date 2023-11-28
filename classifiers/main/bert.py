#!/usr/bin/env python3

# BERT Classifier
# Christophe Leung
# October 30, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
import sys
sys.path.append('../')

from pathlib import Path
from BERT.src.helpers2 import Helpers2 as hlp, Timer
from BERT.src.config import Constants, Hyperparams, Data
from BERT.src.preproc import Preproc
from BERT.src import classify
import torch, time

# =================================================================================================
# Main Function

def main(logger, model_main, df_training, **kwargs):
    evalonly = kwargs['evalonly'] if 'evalonly' in kwargs else False
    hyparams = kwargs['hyparams'] if 'hyparams' in kwargs else {}
    df_testing = kwargs['predict'] if 'predict' in kwargs else None
    mode = kwargs['mode'] if 'mode' in kwargs else Hyperparams.train_mode
    fpath_modelcached = Constants.DPATH_MODELS/Constants.FPATH_MODEL

    hlp.seed_everything()
    train_set, val_set, test_labels, test_goldlabels, test_texts = Preproc.training_split_and_tensorify(df_training, mode=mode)
    
    # ------------------------------------------------------------
    # print train-/test-split stats
    Preproc.tvs_stats(logger, train_set, val_set, test_texts)
    
    if not evalonly:
        # ------------------------------------------------------------
        # train the model on the GPU
        hlp.seed_everything()
        model_main.cuda()
        logger.info('Beginning language model training...')
        classify.model_train(logger, model_main, train_set, val_set, **hyparams)

        # save the model (if necessary)
        logger.info('Caching trained language model to "%s"...' % str(fpath_modelcached))
        print('\n')
        classify.model_save(Constants.MODEL, fpath_modelcached)

    # ------------------------------------------------------------
    # error analysis on the trained model - print scores and wrong predictions if available
    logger.info('Model Evaluation...')
    hlp.seed_everything()
    torch.cuda.empty_cache()
    model_main.load_state_dict(torch.load(fpath_modelcached))
    model_main.cuda()
    errors = classify.model_error(logger, test_texts, test_labels, hyparams=hyparams, n_err=5)
    print()
    logger.info('Examples of Incorrect Predictions...')
    for incorrect in errors:
        print('- '*25)
        logger.info(incorrect)
    print()

    # ------------------------------------------------------------
    # perform the final prediction on the unlabeled test set
    if df_testing:
        hlp.separator(msg='Final Prediction on Unlabeled Test Set\n')
        hlp.seed_everything()
        torch.cuda.empty_cache()
        model_main.load_state_dict(torch.load(Constants.DPATH_MODEL/'model_cached.torch'))
        model_main.cuda()
        predictions_final = classify.predict_final(df_testing, hyparams=hyparams)
        classify.output_final(predictions_final, Constants.DPATH_DATA/'predictions.csv')
        print('\n')


#  ----------------------------------------------------------------------------
# Run the Main Function
hlp.seed_everything()
Constants.GPU_DEVICE = hlp.gpu_init(showmsg=True)[0]
Constants.MODEL = Hyperparams.model_Bert
Data = Data()

if __name__ == '__main__':
    timer, logger = Timer(), hlp.logging_init()
    fpath_training_custom = list(filter(lambda arg: arg.endswith('.csv'), sys.argv[1:]))
    timer.start()
    logger.info('Program started at %s' % timer.start)
    if len(fpath_training_custom) > 0:
        logger.info('Using "%s" as the training data instead...' % fpath_training_custom[0])
        print()
        Data.init(fpath_training_custom[0])
    if '--evalonly' in sys.argv:
        logger.info('Running in test-only (non-training) mode...')
        print()
        main(logger, Constants.MODEL, Data.training, hyparams=Hyperparams.hyperparams, evalonly=True)
    else:
        logger.info('Running in full (train/validate/test) mode...')
        logger.info('Unsaved models will be overwritten. Use Control-C to terminate immediately...')
        print()
        time.sleep(1)
        main(logger, Constants.MODEL, Data.training, hyparams=Hyperparams.hyperparams)
    hlp.separator()
    timer.finish()
    logger.info('Program finished at %s' % timer.finish)
    logger.info('Total Runtime: %.3fs  (%.0fm %.0fs)\n\n' % (timer.total, timer.total//60, timer.total%60))

