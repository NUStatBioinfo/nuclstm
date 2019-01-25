import numpy as np
from pandas import DataFrame


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


def pad_binary_signal(x, pad_len=10):
    """
    Given a binary vector, pad all instances of 1's with pad_len 1's on either side to create
    blocks of >= 2*pad_len + 1 consecutive 1's.

    :param x: binary 0/1 list or np.array
    :param pad_len: int size of pad on a size of a 1
    :return: a 0/1 np.array
    """
    n = len(x)
    one_idx = np.arange(n)[x == 1]

    if len(one_idx) == 0:
        return x

    y = np.zeros(n)
    for idx in one_idx:
        start = max(idx - pad_len, 0)
        end = min(idx + pad_len + 1, n)
        y[start:end] = 1.0

    return y


def get_ksensitivity(df):
    """
    Compare the sensitivities of two nucleosome center prediction methods
    by calling a "successful" location when a predicted ncp is
     +/- k-base pairs distance of some ground truth 'nucleosome' vector.

     See Figure 3 in Wang, Widom (2010).

    :param df: pandas.DataFrame with a 'nucleosome' field along with 'nupop_ncp' and 'nuclstm_ncp' fields.
    :return: pandas.DataFrame with 'k,' 'nupop_tpr,' and 'nuclstm_tpr' sensitivity analysis vectors as a
    function of +/- k base positions away from the nucleosome centers.
    """
    sub_df = df[['nupop_ncp', 'nuclstm_ncp', 'nucleosome']]
    n_nucs = sub_df['nucleosome'].sum()

    pad_lens = list(np.arange(5, stop=75, step=5))
    pad_lens.append(73)
    nupop_tpr = []
    nuclstm_tpr = []

    for pad_len in pad_lens:
        sub_df.loc[:, 'nucleosome_padded'] = pad_binary_signal(sub_df['nucleosome']
                                                               , pad_len=pad_len)

        tpr = ((sub_df['nupop_ncp'] == 1) & (sub_df['nucleosome_padded'] == 1)).sum() / n_nucs
        nupop_tpr.append(100 * tpr)

        tpr = ((sub_df['nuclstm_ncp'] == 1) & (sub_df['nucleosome_padded'] == 1)).sum() / n_nucs
        nuclstm_tpr.append(100 * tpr)

    return DataFrame({'k': pad_lens
                      , 'nupop_tpr': nupop_tpr
                      , 'nuclstm_tpr': nuclstm_tpr})


def get_latin_chrom(idx=None):
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
    return mapping[idx] if idx else mapping
