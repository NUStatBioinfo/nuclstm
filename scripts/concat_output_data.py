#!/bin/python

# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
# Script for concatenating output of test chromosomes into a single .txt
# file for .wig file processing.
#
# E.g. -
# $ python concat_output_data.py -d ../../model_output/gru_binary_sequence_mmddyy_HMS
# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#

import logging
import sys
import os
import re
import argparse
import pandas as pd


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
    parser.add_argument('-f'
                        , '--fields'
                        , nargs='+'
                        , type=str
                        , default=['nuclstm_preds_smoothed', 'nuclstm_preds', 'nuclstm_ncp']
                        , required=False
                        , help='list of fields from test_data_chr*.csv files to concatenate into long files')

    args = parser.parse_args()

    chr_files = list()
    for f in os.listdir(args.data_dir):
        if re.search(r'test_data_chr[0-9]{1,2}.csv', f):
            chr_files.append(f)

    if not chr_files:
        logging.info('No chromosome test data files found in {0}'.format(args.data_dir))
        sys.exit()

    else:
        logging.info('Found output files for {0} chromosomes'.format(len(chr_files)))


    fields = list(set(args.fields))
    chrom_field_dict = dict()

    for f in chr_files:
        chr_filepath = os.path.join(args.data_dir, f)
        logging.info('Loading file {0}'.format(f))
        df = pd.read_csv(os.path.join(chr_filepath))
        chrom = df['Chr'].unique()[0]
        chrom_field_dict[chrom] = dict()

        for field in fields:
            cat_fields = ['Chr', 'pos', field]

            missing_cols = list(set(cat_fields) - set(df.columns))
            if missing_cols:
                raise ValueError('Not all requested fields found in file {0}'.format(chr_filepath))

            chrom_field_dict[chrom][field] = df[cat_fields]

    chroms = list(chrom_field_dict.keys())

    for field in fields:
        for chrom in chroms:
            if chrom == chroms[0]:
                df = chrom_field_dict[chrom][field]
            else:
                df = pd.concat([df, chrom_field_dict[chrom][field]])

        logging.info('---- Field {0} ----'.format(field.upper()))
        logging.info('total concatentated pandas.DataFrame shape: {0}'.format(df.shape))
        logging.info('total number of unique chromosomes: {0}'.format(df['Chr'].nunique()))
        logging.info('\n{0}'.format(df.head(5)))

        out_file = os.path.join(args.data_dir, '{0}_long.txt'.format(field))
        logging.info('Saving long-form data to {0}'.format(out_file))

        df.to_csv(out_file
                  , sep=' '
                  , header=False
                  , index=False)