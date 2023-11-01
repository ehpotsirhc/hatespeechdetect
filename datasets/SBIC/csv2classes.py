#!/usr/bin/env python3

# SBIC Data Post-Processing CSV Column Splitter
# Christophe Leung
# October 29, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# =============================================================================
from pathlib import Path
import pandas as pd, sys

# -----------------------------------------------------------------------------

def main(fname_csv_input, **kwargs):
    fname_csv_input = Path(fname_csv_input)
    column_name = str(kwargs['column_name']) if 'column_name' in kwargs else 'target_minority'
    classes = sorted(pd.read_csv(fname_csv_input)[column_name].unique())
    prefix = fname_csv_input.stem[:-10] if '_processed' in fname_csv_input.stem else fname_csv_input.stem
    pd.DataFrame(classes).to_csv('%s_classes.txt' % (prefix), header=None, index=None)

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        main(sys.argv[1], column_name=sys.argv[2])
    elif len(sys.argv) == 3:
        main(sys.argv)
    else:
        print('  To run, use...')
        print('    python csv2classes.py <input_csv_filename> <column_to_extract>\n')
        print('  Example...')
        print('    python csv2classes.py SBIC.v2.agg.cmb_processed.csv target_minority\n')
