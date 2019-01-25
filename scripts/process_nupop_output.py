import pandas as pd
import re
import os
import sys
import numpy as np
import logging
sys.path.append(os.getenv('NUCLSTM_SOURCE'))
from utils import group_consecutive_ints


def get_latin_chrom(roman_idx):
    mapping = {'chrI': 1,
               'chrII': 2,
                'chrIII': 3,
                'chrIV': 4,
                'chrV': 5,
                'chrVI': 6,
                'chrVII': 7,
                'chrVIII': 8,
                'chrIX': 9,
                'chrX': 10,
                'chrXI': 11,
                'chrXII': 12,
                'chrXIII': 13,
                'chrXIV': 14,
                'chrXV': 15,
                'chrXVI': 16,
                'chrXVII': 17,
                'chrXVIII': 18,
                'chrXIX': 19,
                'chrXX': 20}
    return mapping[roman_idx]

def process_nupop_output(data_dir, output_dir):

    if not os.path.isdir(data_dir):
        raise ValueError('{0} is not a valid directory'.format(data_dir))

    if not os.path.isdir(output_dir):
        raise ValueError('{0} is not a valid directory'.format(output_dir))

    nupop_files = [f for f in os.listdir(data_dir) if re.search(r'Prediction(.*).txt$', f)]
    logging.info('Found {0} nupop prediction files.'.format(len(nupop_files)))

    for f in nupop_files:
        roman_chr = f.split('.')[0]
        idx = get_latin_chrom(roman_chr)

        logging.info('Reading in nupop data from {0}.'.format(f))
        with open(os.path.join(data_dir, f), 'r') as dat:
            nupop_dat = dat.read()

        lines = [x.split() for x in nupop_dat.split('\n')]
        cols = lines[0]
        cols[0] = 'pos'
        lines = lines[1:]

        # remove empty list elements.
        lines = list(filter(None, lines))

        # cast to pd.DataFrame.
        df = pd.DataFrame(lines
                          , columns=cols)

        # determine the predicted nucleosome centers from the Viterbi output.
        df['N/L'] = df['N/L'].astype(int)
        nuc_occ_groups = group_consecutive_ints(np.where(df['N/L'].values == 1)[0])
        nuc_center_idx = list(map(lambda x: x[74], nuc_occ_groups))

        df.loc[:, 'nupop_ncp'] = 0
        df.loc[nuc_center_idx, 'nupop_ncp'] = 1

        # append chromosome identifier
        df['Chr'] = 'chr' + str(idx)
        df.reset_index(inplace=True)
        if 'index' in df.columns:
            df.drop('index'
                    , axis=1
                    , inplace=True)

        out_file = 'chr_{0}_nupop.feather'.format(idx)
        logging.info('Saving nupop data for chromosome {0} to {1}.'.format(idx, out_file))
        df.to_feather(os.path.join(output_dir, out_file))


if __name__ == '__main__':
    log_dir = os.path.dirname(os.path.realpath(__file__))
    logging.basicConfig(handlers=[logging.StreamHandler()]
                        , level=logging.DEBUG
                        , format='%(asctime)s ---- %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

    data_dir = os.path.join(os.getenv('HOME')
                            , 'nu'
                            , 'jiping_research'
                            , 'data'
                            , 'nupop_output')

    process_nupop_output(data_dir
                         , output_dir=data_dir)