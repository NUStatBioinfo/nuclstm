import argparse
import os
import numpy as np
from multiprocessing import cpu_count
from pandas import DataFrame, concat


def max_cpu():
    """
    :return: int number of CPUs available on a node minus 1
    """
    return cpu_count() - 1

def check_dataframe_validity(df, reqd_cols=None):
    """
    Check that a pandas.DataFrame is not empty and has various required columns.

    :param df: pandas.DataFrame
    :param reqd_cols: list or str, names of columns that must be within df
    :return: True if df checks out
    """

    if reqd_cols and not isinstance(reqd_cols, list):
        reqd_cols = [reqd_cols]

    # ensure that df is actually a pandas.DataFrame
    if not isinstance(df, DataFrame):
        raise ValueError('input data is not a pandas.DataFrame object.')

    # ensure the data actually has data.
    if df.empty:
        raise ValueError('input data is empty!')

    # ensure that all reqd_cols are columns in df
    if reqd_cols and not all([x in df.columns for x in reqd_cols]):
        raise ValueError('input DataFrame does not contain all of the required columns: {0}'
                         .format(', '.join(reqd_cols)))

    return True


def split_comma_ints(s):
    return list(map(lambda x: int(x.strip()), s.split(',')))


def group_consecutive_ints(x):
    """
    Return list of consecutive lists of integers from a list or np.array.
    Taken from https://bit.ly/2wrWxDO

    :param x: list or np.array of ints
    :return: list of lists of grouped consecutive values in x
    """
    run = []
    result = [run]
    expect = None
    for v in x:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + 1

    return result


def parse_nuc_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i'
                        , '--input_file'
                        , type=str
                        , default=None
                        , required=True
                        , help='name of file with nucleosome positioning data joined with genome sequence')
    parser.add_argument('-f'
                        , '--config_file'
                        , type=str
                        , default=None
                        , required=True
                        , help='name of model config .json file')
    parser.add_argument('-o'
                        , '--output_dir'
                        , type=str
                        , default=None
                        , required=True
                        , help='realpath of an output directory for storing model progress, trained models, etc.')
    parser.add_argument('-c'
                        , '--train_chromosomes'
                        , type=str
                        , default='1,2,3,4'
                        , help='comma-separated string of training-set yeast chromosomes')
    parser.add_argument('-v'
                        , '--val_chromosomes'
                        , type=str
                        , default='5'
                        , help='comma-separated string of validation-set yeast chromosomes')
    parser.add_argument('-t'
                        , '--test_chromosomes'
                        , type=str
                        , default='6,7'
                        , help='comma-separated string of test-set yeast chromosomes')
    parser.add_argument('-m'
                        , '--pretrained_model'
                        , type=str
                        , default=None
                        , help='name of an h5 file containing a pre-trained Keras model')
    parser.add_argument('-n'
                        , '--nupop_dir'
                        , type=str
                        , default=None
                        , help='path to a directory containing NuPoP predictions, for comparison with nuclstm.')

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise IOError('input_file not found')

    if not os.path.isfile(args.config_file):
        raise IOError('config_file not found')

    if args.nupop_dir:
        if not os.path.isdir(args.nupop_dir):
            raise IOError('nupop_dir not found')

    if args.pretrained_model:
        if not os.path.isfile(args.pretrained_model):
            raise IOError('pretrained_model not found')
        if not (args.pretrained_model.endswith('.h5') or args.pretrained_model.endswith('.hdf5')):
            raise ValueError('pretrained model file must be an .h5 or .hdf5 file.')

    args.train_chromosomes = split_comma_ints(args.train_chromosomes)
    args.val_chromosomes = split_comma_ints(args.val_chromosomes)
    args.test_chromosomes = split_comma_ints(args.test_chromosomes)

    return args