import os
import sys
import pandas as pd
import argparse
import logging
import re

sys.path.append(os.getenv('NUCLSTM_SOURCE'))
from evaluation import plot_cross_correlation


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

    for f in chr_files:
        chr_filepath = os.path.join(args.data_dir, f)
        df = pd.read_csv(os.path.join(chr_filepath))
        chrom = df['Chr'].unique()[0]

        logging.info('Generating cross-correlation plot for chromosome {0}'.format(chrom))
        plot_cross_correlation(df['nucleosome'].values
                               , preds=df['nuclstm_ncp'].values
                               , fname=os.path.join(args.data_dir,
                                                    'cross_correlation_' + str(chrom) + '.png')
                               , lag_width=80
                               , figsize=(8, 6))