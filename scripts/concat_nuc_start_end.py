#!/bin/python

# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
# Script for concatenating nucleosome occupation calls from output of test
# chromosomes into a single .txt file for .wig file processing.
#
# E.g. -
# $ python concat_nuc_start_end.py -d ../../model_output/gru_binary_sequence_mmddyy_HMS
# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#

import logging
import sys
import os
import re
import argparse
import pandas as pd
import numpy as np


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout
                        , level=logging.DEBUG
                        , format='%(asctime)s ---- %(message)s'
                        , datefmt='%m/%d/%Y %I:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d'
                        , '--data_dir'
                        , type=str
                        , default=None
                        , required=True
                        , help='realpath of a directory containing model output data.')

    args = parser.parse_args()

    chr_files = list()
    for f in os.listdir(args.data_dir):
        if re.search(r'test_data_chr_[0-9]{1,2}.csv', f):
            chr_files.append(f)

    if not chr_files:
        logging.info('No chromosome test data files found in {0}'.format(args.data_dir))
        sys.exit()

    else:
        logging.info('Found output files for {0} chromosomes'.format(len(chr_files)))

    sub_fields = ['Chr', 'start', 'end']
    out_df = None

    for f in chr_files:
        chr_filepath = os.path.join(args.data_dir, f)
        logging.info('Loading file {0}'.format(f))
        df = pd.read_csv(os.path.join(chr_filepath))
        max_pos = df.pos.max()

        # subset to nuc centers, outline nucleosome starts/ends
        df = df[df['nuclstm_ncp'] == 1]
        df['start'] = df['pos'] - 73
        df['end'] = df['pos'] + 73

        # make sure no nucleosome starts are < 1 or > length of chromosome
        left_bound_idx = list(np.where(df.start < 1)[0])
        right_bound_idx = list(np.where(df.end > max_pos)[0])

        if left_bound_idx:
            df.start.iloc[list(left_bound_idx)] = 1

        if right_bound_idx:
            df.end.iloc[list(right_bound_idx)] = max_pos

        if not isinstance(out_df, pd.core.frame.DataFrame):
            out_df = df[sub_fields]

        else:
            out_df = pd.concat([out_df
                                   , df[sub_fields]])

    logging.info('total concatentated pandas.DataFrame shape: {0}'.format(out_df.shape))
    logging.info('total number of unique chromosomes: {0}'.format(out_df['Chr'].nunique()))
    logging.info('\n{0}'.format(out_df.head(5)))

    out_file = os.path.join(args.data_dir, 'nuclstm_nuc_occup_long.txt')
    logging.info('Saving long-form nucleosome occupancy data to {0}'.format(out_file))

    out_df.to_csv(out_file
                  , sep=' '
                  , header=False
                  , index=False)