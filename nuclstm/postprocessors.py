from utils import *
from preprocessors import preprocess_by_chromosome
from scipy.signal import argrelmax
from joblib import Parallel, delayed


def get_max_dist_blocks(x, max_dist=127):
    """
    Group together elements of a list or vector based on maximum pairwise distance between
    the elements. E.g. with a max_dist of 5, the maximum that any two elements in a group
    will differ will be < 5.

    E.g. [1, 2, 4, 8, 9, 10] with max_dist of 3 will be broken into
    [[1, 2, 4], [8, 9, 10]].

    Taken from https://bit.ly/2jJKPvl

    :param x: list or np.array
    :param max_dist: distance between any two elements in the same group
    :return: list of lists of grouped elements of x
    """
    mi, ma = 0, 0
    result = []
    temp = []
    for v in sorted(x):
        if not temp:
            mi = ma = v
            temp.append(v)
        else:
            if abs(v - mi) < max_dist and abs(v - ma) < max_dist:
                temp.append(v)
                if v < mi:
                    mi = v
                elif v > ma:
                    ma = v
            else:
                result.append(temp)
                mi = ma = v
                temp = [v]

    return result


def find_local_maximums_with_seps(x, min_sep=127, return_binary=True):
    """
    Find local maximums of a vector that are separated by at least a certain distance. If multiple peaks
    occur within min_sep of each other, take the highest peak.

    E.g. x=[0, 1, 2, 1, 5, 4, 3, 1, 1, 10, 5] with a min_sep=3 would return [4, 9].

    :param x: data vector with supposed peaks
    :param min_sep: minimum separation (in indices) of local peaks
    :param return_binary: boolean, would you like output to be a vector of 0/1 values indicating local maximum?
    :return: indices of local peaks if return_binary is False, else return vector of 0/1's with length len(x)
    indicating presence of local maximum
    """
    # identify local maxima.
    peak_idx = argrelmax(x)[0]

    if len(peak_idx) == 0:
        return None

    # group local maxima together based on maximum pairwise distance.
    peak_groups = get_max_dist_blocks(peak_idx
                                      , max_dist=min_sep)
    sep_peak_idx = list(map(lambda y: y[np.argmax(x[y])], peak_groups))

    # if return_binary, turn set of local maxima into vector of 0/1 indicator of maxima.
    if return_binary:
        return [1 if i in sep_peak_idx else 0 for i in range(len(x))]

    return sep_peak_idx


def locate_nucleosome_centers(df, pred_col='preds', min_sep=127, min_max=0.5, n_jobs=max_cpu()):
    """
    For each chromosome, locate predicted nucleosome centers from a vector of NCP/probability scores,
    also append the smoothed NCP/probability score used to find local peaks in scores.

    :param df: pandas.DataFrame with at least 'Chr' and `pred_col` columns.
    :param pred_col: name of column in `df` corresponding to ncp predictions
    :param min_sep: int minimum distance allowed between two predicted nucleosome center positions
    :param min_max: float minimum value that a local peak must obtain in order to be considered a local maximum;
    applied when finding local maxima of smoothed predictions vector.
    :param n_jobs: int number of CPUs for parallelizing nucleosome center search algorithm over chromosomes.
    :return: altered df now containing 'pred_ncp' field with 0/1 data indicating presence
    of predicted nucleosome center.
    """
    check = check_dataframe_validity(df
                                     , reqd_cols=['Chr', pred_col])

    smoothed_col = pred_col + '_smoothed'

    # Smooth the predicted probabilities with a gaussian.
    df = preprocess_by_chromosome(df
                                  , target=pred_col
                                  , method='smooth'
                                  , smooth_window_len=41  # why is this 41?
                                  , smooth_window='hanning')

    # identify chromosomes, groupby them.
    chroms = df['Chr'].unique()
    g = df.groupby('Chr')

    # apply, by chromosome, the greedy peak finder.
    peak_df = g.apply(lambda x: find_local_maximums_with_seps(x[smoothed_col].values
                                                              , min_sep=min_sep
                                                              , return_binary=True))
    # par_output = Parallel(n_jobs=min(n_jobs, len(chroms))
    #                       , verbose=0
    #                       , backend='threading')(delayed(find_local_maximums_with_seps)(
    #     x=df[df.Chr == chrom][smoothed_col].values
    #     , min_sep=min_sep
    #     , return_binary=True)
    #     for chrom in chroms)

    # append binary 0/1 indicating predicted center of nucleosome found from peak detection.
    # append smoothed predictions.
    df.loc[:, 'nuclstm_ncp'] = 0
    for i in range(len(chroms)):
        chrom = chroms[i]
        chrom_idx = np.where(df['Chr'] == chrom)[0]
        # df.iloc[chrom_idx, df.columns.get_loc('nuclstm_ncp')] = par_output[i]
        df.iloc[chrom_idx, df.columns.get_loc('nuclstm_ncp')] = peak_df.loc[chrom]

    # only keep ncp's where the max predicted ncp probability is above min_max threshold.
    df['nuclstm_ncp'] = ((df['nuclstm_ncp'] == 1) & (df[pred_col].rolling(147
                                                                          , min_periods=1).max() >= min_max)).astype(int)

    return df


def get_minimum_binary_distance(x, y):
    """
    For two binary vectors, find the distances between every 1 in x and the nearest 1 in y.

    For example:
    x = [0, 1, 0, 0, 0, 1]
    y = [1, 1, 0, 0, 1, 0]
    results in [(1-1 = 0), (5-4 = 1)]

    :param x: np.array of 0's and 1's
    :param y: np.array of 0's and 1's
    :return: np.array of minimum distances
    """
    dists = np.zeros(len(x))
    for i in range(len(x)):
        dists[i] = np.min(np.abs(x[i] - y))

    return dists