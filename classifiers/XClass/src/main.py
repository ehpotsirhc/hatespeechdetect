#!/usr/bin/env python3

# XClass Classifier
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from config import Constants, Hyperparams, Data
import sys, time, argparse
import utils



# =================================================================================================
# Main Function

def main(args, logger):
    logger.info('Main function started'), logger.info(args)
    if args.dataset and args.dataset.endswith('.csv'):
        Data.init(args.dataset)

    
    # train_set, val_set, test_labels, test_texts = Preproc.training_split_and_tensorify(df_training)
    # print(len(df_training))


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

