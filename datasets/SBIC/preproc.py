# SBIC Data Pre-Processing
# Christophe Leung, Iris Chen, Kelvin Lu
# October 29, 2023

# This file is a duplicate of the Jupyter Notebook for compatibility reasons.
# If Jupyter is available, please use the Jupyter Notebook instead.
# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# =============================================================================
from pathlib import Path
import numpy as np, pandas as pd, re

FPATH_SBIC = Path('./')

# -----------------------------------------------------------------------------
# Original Data Import

train_raw = pd.read_csv(FPATH_SBIC/'SBIC.v2.agg.trn.csv')
train_raw


# -----------------------------------------------------------------------------
# Filter for Target Classes

targets = ['women', 'black', 'asian', 'jewish', 'muslim', 'gay', 'lesbian', 'trans', 'bisexual', 'asexual']

target_minority = []
def has_class(targetMinority):
    # return len(  set(targets).intersection( set(eval(targetMinority)) )  ) > 0
    for target in targets:
        if target in targetMinority.lower():
            target = 'LGBT' if target in ['gay', 'lesbian', 'trans', 'bisexual', 'asexual'] else target
            target_minority.append(target)
            return True
    return False

train_filtered = train_raw[ train_raw.targetMinority.apply(has_class) ]
train_preproc = pd.DataFrame()
train_preproc['post'] = train_filtered.post
train_preproc['target_minority'] = target_minority
train_preproc = train_preproc.reset_index(drop=True)

print(train_preproc.target_minority.value_counts())
train_preproc


# -----------------------------------------------------------------------------
# Pre-Process Data + Add Integer Labels
def textproc(text):
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

classes = list(train_preproc.target_minority.unique())
labels = [classes.index(c) for c in train_preproc.target_minority]

train_postproc = pd.DataFrame()
train_postproc = train_preproc.copy()

train_postproc.post = train_postproc.post.apply(textproc)
train_postproc['labels'] = labels
train_postproc = train_postproc.dropna()


print(classes)
train_postproc


# -----------------------------------------------------------------------------
# Output to Required Format
# Uncomment to save/overwrite the existing data

# train_postproc.post.to_csv('dataset.txt', header=None, index=None)
# train_postproc.labels.to_csv('labels.txt', header=None, index=None)
# np.savetxt('classes.txt', classes, fmt='%s')


# -----------------------------------------------------------------------------
# Compare Pre-Proc and Post-Proc (Check Processing)
# Below we show the removal of @mentions, &entities, links, linebreak characters, linefeed characters, emojis as well as blank(NaN) lines

# Pre-Processed Posts
pd.set_option('max_colwidth', 150)
print('Total rows in PRE-processed DataFrame: %s' % len(train_preproc))
train_preproc.loc[[1, 13, 14, 19, 39, 180, 3290, 4552, 3282, 4518], :]

# Post-Processed Posts
print('Total rows in POST-processed DataFrame: %s' % len(train_postproc))
train_postproc.loc[[1, 13, 14, 19, 39, 180, 3290, 4552, 3282, 4518], :]

