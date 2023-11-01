#!/usr/bin/env python3

# SBIC Data Pre-Processing
# Christophe Leung, Iris Chen, Kelvin Lu
# October 29, 2023

# This file serves as the primary preprocessing script.
# The Jupyter Notebook serves as a testbed for data exploration.
# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# =============================================================================
from pathlib import Path
import numpy as np, pandas as pd, re, sys

# -----------------------------------------------------------------------------
class Helpers:
    def __init__(self):
        self.target_minority = []

    def has_class(self, targetMinority):
        targets = ['women', 'black', 'asian', 'jewish', 'muslim', 'gay', 'lesbian', 'trans', 'bisexual', 'asexual']
        lgbt = ['gay', 'lesbian', 'trans', 'bisexual', 'asexual']
        for target in targets:
            if target in targetMinority.lower():
                target = 'LGBT' if target in lgbt else target
                self.target_minority.append(target)
                return True
        return False
    
    def textproc(self, text):
        urls = r'https*://.+?[\s\n]*$'              # remove urls
        mentions = r'@.+?\s'                        # remove @mentions
        misc = r'&#\d+;|&.+?;'                      # remove miscellaneous items such as &entities
        chars = r'[\r\n]'                           # remove line breaks
        exclude = r'[^a-zA-Z0-9\s\'\.\,\;\!\-\@\#]' # keep these characters
        processed = re.sub(urls, 'http ', text).strip()
        processed = re.sub(mentions, '@user ', processed).strip()
        processed = re.sub(r'(%s|%s|%s)'% (misc, chars, exclude), '', processed).strip()
        processed = np.NaN if len(processed.replace(' ', ''))==0 else processed
        return processed

    def summary(self, df_raw, df_filtered, df_preproc, df_postproc):
        fmt_digits = 8
        print('Summary Stats')
        print('Records Processed... %s' % str(len(df_raw)).rjust(fmt_digits))
        print('Filtered Records...  %s' % str(len(df_filtered)).rjust(fmt_digits))
        print('Pre-processed...     %s' % str(len(df_preproc)).rjust(fmt_digits))
        print('Post-processed...    %s' % str(len(df_postproc)).rjust(fmt_digits))
        print()
        print('Class Distribution:')
        print(df_preproc.target_minority.value_counts(), '\n')


def main(fpath_data):
    fpath_data = Path(fpath_data)                                               # sanitize CLI input
    hlp = Helpers()                                                             # instantiate the Helpers class

    train_raw = pd.read_csv(fpath_data)                                         # import the data
    train_filtered = train_raw[ train_raw.targetMinority.apply(hlp.has_class) ] # filter targetMinority for the classes we want
    
    train_preproc = pd.DataFrame()                                              # create the pre-processing DataFrame
    train_preproc['post'] = train_filtered.post                                 # add in the "post" column
    train_preproc['target_minority'] = hlp.target_minority                      # add in the "target_minority" column
    train_preproc = train_preproc.reset_index(drop=True)                        # reset the DataFrame index given the current curation

    classes = sorted(train_preproc.target_minority.unique())                    # the unique class labels
    labels = [classes.index(c) for c in train_preproc.target_minority]          # the class labels as an integer (required by XClass)

    train_postproc = pd.DataFrame()                                             # create the post-processing DataFrame
    train_postproc = train_preproc.copy()                                       # copy over the preprocessed data as a starting template
    train_postproc.post = train_postproc.post.apply(hlp.textproc)               # apply the preprocessing function
    train_postproc['label'] = labels                                            # add in the labels column
    train_postproc = train_postproc.dropna()                                    # drop null values in the table

    hlp.summary(train_raw, train_filtered, train_preproc, train_postproc)

    # # OUTPUT THE PROCESSED DATA TO DISK
    # # Uncomment the following 3 lines to save/overwrite the existing files
    train_postproc.to_csv('%s_processed.csv' % fpath_data.stem, index=None)                 # full CSV
    train_postproc.post.to_csv('%s_dataset.txt' % fpath_data.stem, header=None, index=None) # list of text posts
    train_postproc.label.to_csv('%s_labels.txt' % fpath_data.stem, header=None, index=None) # list of class labels as integers
    np.savetxt('%s_classes.txt' % fpath_data.stem, classes, fmt='%s')                       # list of classes


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('  To run, use "python preproc.py <datafile>"\n')


