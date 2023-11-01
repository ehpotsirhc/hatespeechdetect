#!/usr/bin/env python3

# SBIC Data Post-Processing CSV Column Splitter
# Christophe Leung
# October 29, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# =============================================================================
from pathlib import Path
import pandas as pd, sys

# -----------------------------------------------------------------------------

def main(fname_csv_input):
    fname_csv_input = Path(fname_csv_input)
    to_split = pd.read_csv(fname_csv_input)
    for col in to_split.columns:
        prefix = fname_csv_input.stem[:-10] if '_processed' in fname_csv_input.stem else fname_csv_input.stem
        to_split[col].to_csv('%s_%s.txt' % (prefix, col), header=None, index=None)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('  To run, use "python csv2txt.py <input_csv_filename_>"\n')

