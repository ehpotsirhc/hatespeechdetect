#!/usr/bin/env python3

# BERT Classifier
# Christophe Leung
# October 30, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from helpers2 import Helpers2 as hlp
from config import Constants, Hyperparams, Data
from preproc import Preproc
import torch, sys, time
import classify

# =================================================================================================
# Main Function

def main(model_main, df_training, **kwargs):
    testing_only = kwargs['testing_only'] if 'testing_only' in kwargs else False
    hyparams = kwargs['hyparams'] if 'hyparams' in kwargs else {}
    df_testing = kwargs['predict'] if 'predict' in kwargs else None

    hlp.seed_everything()
    train_set, val_set, test_labels, test_texts = Preproc.training_split_and_tensorify(df_training)
    
    # ------------------------------------------------------------
    # print train-/test-split stats
    Preproc.tvs_stats(train_set, val_set, test_texts)
    
    if not testing_only:
        # ------------------------------------------------------------
        # train the model on the GPU
        hlp.seed_everything()
        model_main.cuda()
        classify.model_train(model_main, train_set, val_set, **hyparams)

        # save the model (if necessary)
        classify.model_save(Constants.MODEL, Constants.DPATH_MODEL/'model_cached.torch')

    # ------------------------------------------------------------
    # error analysis on the trained model - print accuracy and 5 wrong predictions if available
    hlp.separator(msg='Model Accuracy & Error Analysis - print 5 incorrect predictions if available\n')
    hlp.seed_everything()
    torch.cuda.empty_cache()
    model_main.load_state_dict(torch.load(Constants.DPATH_MODEL/'model_cached.torch'))
    model_main.cuda()
    errors = classify.model_error(test_texts, test_labels, hyparams=hyparams, n_err=5)
    print('\nExamples of Incorrect Predictions...')
    [print('- '*25, '\n', e) for e in errors]
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
    fpath_training_custom = list(filter(lambda arg: arg.endswith('.csv'), sys.argv[1:]))
    if len(fpath_training_custom) > 0:
        print('Using "%s" as the training data instead...\n' % fpath_training_custom[0])
        Data.init(fpath_training_custom[0])
    if '--testing-only' in sys.argv:
        print('Running in test-only (non-training) mode...\n')
        main(Constants.MODEL, Data.training, hyparams=Hyperparams.hyperparams, testing_only=True)
    else:
        print('Running in full (train/validate/test) mode...')
        print('Unsaved models will be overwritten. Use Control-C to terminate immediately...\n')
        time.sleep(1)
        main(Constants.MODEL, Data.training, hyparams=Hyperparams.hyperparams)
