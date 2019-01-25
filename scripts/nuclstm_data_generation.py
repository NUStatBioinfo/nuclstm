#!/bin/python

# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
# Script for creating full set of mouse transcript nucleosome position + sequence data
# Specify 'unique' or 'redundant' to build either unique or redundant file.
#
# Note: unique map is available at https://media.nature.com/original/nature-assets/nature/journal/v486/n7404/extref/nature11142-s2.txt
# and redundant map is available at https://media.nature.com/original/nature-assets/nature/journal/v486/n7404/extref/nature11142-s3.txt
#
#
# Create a DataFrame resembling
#
# E.g. -
# $ python concat_output_data.py -d ../../model_output/gru_binary_sequence_mmddyy_HMS
# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#

import os
import sys
import re
import logging
import pandas as pd
from Bio import SeqIO
import argparse


def main():
    """
    nucleosome maps used to train LSTM models are ust the Nature supplement data nucleosome positions
    joined with fasta sequences, joined with NCP score data `NNT_cutWC.NCP.Ratio.txt`.

    Create the full nucleosome / seq / position dataset resembling
    +---------+------+------+-------+-------+-------------+--------------+
    |   Chr   |  pos | ...  |  seq  |  NCP  |  NCP/noise  | nucleosome   |
    +=========+======+======+=======+=======+=============+==============+
    | chr11   |  34  | ...  |   A   | 1.84  |    3.22     |       1      |
    +---------+------+------+-------+-------+-------------+--------------+
    | chr11   |  35  | ...  |   C   | 0.67  |    0.41     |       0      |
    +---------+------+------+-------+-------+-------------+--------------+

    Create a full redundant or unique map dataset, full with nucleosome center data, sequence data, and NCP scores,
    and save it.
    """
    logging.basicConfig(stream=sys.stdout
                            , level=logging.DEBUG
                            , format='%(asctime)s ---- %(message)s'
                            , datefmt='%m/%d/%Y %I:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m'
                        , '--map'
                        , type=str
                        , default=None
                        , required=True
                        , help='nucleosome map to construct. Must be one of their "unique" or "redundant"')
    parser.add_argument('-o'
                        , '--output-dir'
                        , type=str
                        , default='.'
                        , required=False
                        , help='path to directory where output nucleosome map should be saved')

    args = parser.parse_args()

    # specified argument QA
    map_file = args.map.lower()
    if map_file not in ['unique', 'redundant']:
        raise ValueError('--map must be one of "unique" or "redundant"!')

    if not os.path.exists(args.output_dir):
        raise NotADirectoryError('--output-dir not a directory')

    # where I keep nuclstm data
    raw_data_dir = '/Users/fineiskid/nu/jiping_research/data'

    # define file we're going save.
    output_file = os.path.join(args.output_dir
                               , '{0}_nucleosome_map.feather'.format('redundant' if map_file == 'redundant' else 'unique'))

    # -------------------------------------------------------------------------------------#
    # -------------------------------------------------------------------------------------#
    # Build dataset:
    # 1. Load Nature supplement data
    # 2. Load chromosome fasta files, construct sequence DataFrame.
    # 3. Load NCP data.
    # 4. Left join 2. with 3. - left join seq data on NCP data (will cause NAs)
    # 5. Left join 4. with 1. - now we have sequences, NCP scores, and nucleosome centers.
    # -------------------------------------------------------------------------------------#
    # -------------------------------------------------------------------------------------#
    logging.info('Creating full {0} map dataset.'.format(map_file.upper()))

    # Step 1.
    supp_file = 'NIHMS370046-supplement-{0}.txt'.format('4' if map_file == 'redundant' else '5')

    logging.info('Loading Nature supplement file {0} ({1} map)'.format(supp_file, map_file))
    nature_nuc_df = pd.read_table(os.path.join(raw_data_dir, supp_file)
                                  , sep='\s+'
                                  , names=['Chr', 'pos', 'NCP', 'NCP/noise'])

    nature_nuc_df = nature_nuc_df[['Chr', 'pos']].copy()
    nature_nuc_df['nucleosome'] = 1.

    # Step 2.
    logging.info('Loading/creating genome sequence DataFrame')
    fasta_dir = os.path.join(raw_data_dir, 'chromFa')
    seq_dfs = list()
    for f in os.listdir(fasta_dir):
        if re.search('chr', f) is not None:
            seq_dat = SeqIO.read(os.path.join(raw_data_dir, 'chromFa', f), "fasta")
            seq = str(seq_dat.seq)
            chrom = seq_dat.name

            # position is 1-indexed.
            seq_dfs.append(pd.DataFrame({'Chr': chrom
                                         , 'pos': range(1, len(seq) + 1)
                                         , 'seq': [nuc for nuc in seq]}))

    seq_df = pd.concat(seq_dfs)

    # Step 3.
    logging.info('Loading NCP data')
    nnt_df = pd.read_table(os.path.join(raw_data_dir, 'NNT_cutWC.NCP.Ratio.txt')
                           , sep='\s+')

    # Step 4.
    logging.info('Performing left joins')
    df = seq_df.join(nnt_df.set_index(['Chr', 'pos'])
                     , on=['Chr', 'pos']
                     , how='left')

    # Step 5.
    df = df.join(nature_nuc_df.set_index(['Chr', 'pos'])
                 , on=['Chr', 'pos']
                 , how='left')

    # replace NaN nucleosome indicators (non-nuc center locations) with zeros.
    df.loc[df.nucleosome.isnull(), 'nucleosome'] = 0.

    logging.info('data.shape before dropping missing values:{0}'.format(df.shape))
    df.dropna(axis=0, inplace=True)
    logging.info('data.shape after dropping missing values:{0}'.format(df.shape))
    logging.info('Total number of nulceosome centers: {0}'.format(df.nucleosome.sum()))

    # switch Roman numeral chromosome names to numeric.
    chroms = ['chrI', 'chrII', 'chrIII', 'chrIV', 'chrV', 'chrVI', 'chrVII'
        , 'chrVIII', 'chrIX', 'chrX', 'chrXI', 'chrXII', 'chrXIII', 'chrXIV'
        , 'chrXV', 'chrXVI', 'chrXVII', 'chrXVIII', 'chrXIX', 'chrXX', 'chrXXI'
        , 'chrXXII', 'chrXXIII']
    chroms = [x.upper() for x in chroms]
    roman_numeric_map = dict(zip(chroms, [i + 1 for i in range(len(chroms))]))
    
    logging.info('Changing chromosome names from roman numerals to integers')
    df.Chr = df.Chr.apply(lambda x: roman_numeric_map.get(x.upper()))

    # order df by chromosome, position. Reset index.
    df.sort_values(['Chr', 'pos']
                   , axis=0
                   , inplace=True)
    df.reset_index(inplace=True)

    # make chromosome names strings again.
    df.Chr = df.Chr.apply(lambda x: 'chr' + str(x))

    logging.info('final dataset:')
    print(df.head())

    logging.info('shape: {0}'.format(df.shape))
    logging.info('number of nucleosomes: {0}'.format(df.nucleosome.sum()))
    logging.info('number of unique chromosomes: {0}'.format(len(df.Chr.unique())))

    logging.info('saving {0} map data to {1}'.format(map_file, output_file))
    df.to_feather(output_file)


if __name__ == '__main__':
    main()