# SBIC Data Preparation - CSV Stacker
# Christophe Leung, Iris Chen, Kelvin Lu
# October 29, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# =============================================================================
from pathlib import Path
import pandas as pd, sys

# -----------------------------------------------------------------------------

def main(fpath_output, list_fnames_csv):
    fpath_output = Path(fpath_output) if fpath_output.endswith('.csv') else Path('%s.csv' % fpath_output)
    to_combine = [pd.read_csv(Path(fname)) for fname in list_fnames_csv]
    df_combined = pd.concat(to_combine, axis=0).reset_index(drop=True)
    df_combined.to_csv(fpath_output, index=None)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname_output, list_fnames_csv = sys.argv[1], sys.argv[2:]
        main(fname_output, list_fnames_csv)
    else:
        print('  To run, use "python preproc.py <filename_output> <datafile1> <datafile2> <datafileN...>"\n')

