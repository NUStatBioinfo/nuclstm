from utils import *


def GetNucleotideMap():
    nuc = {'A': 0
        , 'C': 1
        , 'T': 2
        , 'G': 3}
    return (nuc)


def encode_sequence(seq, mapping=None):
    """
    Encode a string oligonucleotide sequence into a list of integers,
    e.g. 'ATCCG' into [0, 2, 1, 1, 3].

    :param seq: str single string representation of an oligonucleotide e.g. 'ATCGTCTAC...'
    :param mapping: dict map from nucleotide bases to integers
    :return: list of integers, mapped from nucleotide bases.
    """
    if not mapping:
        mapping = GetNucleotideMap()

    n = len(seq)
    embedding = np.zeros(n
                         , dtype=int)
    for i in range(n):
        if seq[i] in mapping:
            embedding[i] = mapping[seq[i]]
        else:
            raise Exception('%s is not a valid nucleotide!' % i)

    return embedding


def embed_nucleotide_seq(seq):
    """
    One-hot-encode a sequence of nucleotide bases into an n x 4 matrix.

    :param seq: str single string representation of an oligonucleotide e.g. 'ATCGTCTAC...'
    :return: numpy 2-d array of one-hot-encoded bases with columns A, C, T, G (in that order)
    """
    encoded_seq = encode_sequence(seq)
    ohe_mat = np.zeros([len(seq), 4])

    for i in range(len(seq)):
        ohe_mat[i, encoded_seq[i]] = 1.0

    return ohe_mat


def smooth_signal(x, smooth_window_len=11, smooth_window='hamming', scale=False):
    """
    Apply 1-d kernel smoothing to a vector.

    :param x: numpy.array of data to smooth
    :param smooth_window_len: int window length, size of smoother.
    :param smooth_window: str type of smoother. Must be one of `flat`, `hanning`, `hamming`,
    `bartlett`, or `blackman`
    :return: numpy.array of smoothed x data
    """
    if smooth_window_len % 2 == 0:
        raise ValueError('window_len should be odd you dummy.')

    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if x.size < smooth_window_len:
        raise ValueError('Input vector needs to be bigger than window size.')

    if smooth_window_len<3:
        return x

    if not smooth_window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # impute missing values
    x[np.isnan(x)] = 0.0

    # obtain window kernel
    s = np.r_[x[smooth_window_len-1:0:-1], x, x[-2:-smooth_window_len-1:-1]]
    if smooth_window == 'flat':  #moving average
        w = np.ones(smooth_window_len, 'd')
    else:
        w = eval('np.'+smooth_window+'(smooth_window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')
    y = y[(smooth_window_len - 1):(len(y) - smooth_window_len + 1)]

    if scale:
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

    return y


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


def pyramid_proximity_convolution(x, width=53):  # 53 = (107 - 1) / 2 (for unique nucleosome map)
    """
    Convolve pyramid filter with binary input vector, so that we obtain
    proximity to the 1's within the binary input vector, up to a certain distance (past which proximity is 0).
    Output will be a continuously valued vector, use for regression purposes.

    :param x: numpy 1-d binary vector
    :param width: int width of pyramid filter
    :return: numpy 1-d convolved output, scaled between 0 and 1
    """
    # identify center locations
    center_idx = np.where(x == 1)[0]

    # output storage
    conv_vec = np.zeros(len(x))

    for center in center_idx:
        # create pyramid filter around center location
        pyr_left = np.maximum(0, x[x < center] + (width - center))
        pyr_right = np.maximum(0, -1 * x[x > center] + center + width)
        pyr = np.concatenate([pyr_left, np.array([width]), pyr_right])

        # add convolution contribution to output
        conv_vec += pyr

    # scale convolved proximity to between 0 and 1.
    conv_vec = (conv_vec - np.min(conv_vec)) / (np.max(conv_vec) - np.min(conv_vec))

    return conv_vec


def preprocess_by_chromosome(df
                             , target='NCP/noise'
                             , method='standardize'
                             , **kwargs):
    """
    Apply preprocessing functions to a target vector witin a pandas.DataFrame
    on a per-chromosome basis. DataFrame must have a `Chr` column in addition to
    target.

    :param df: pandas.DataFrame with a target field.
    :param target: str name of column in df to apply processing
    :param method: str name of a processing function. Must be one of
    `standardize` ((x - mean)/stddev), `scale` (0-1), and `smooth` (1-d kernel smoothing)
    :param kwargs: named arguments to smooth_signal function if method=='smooth'
    :return: pandas.DataFrame with a new field appended to it with processed-by-chromosome data.
    """

    check = check_dataframe_validity(df
                                     , reqd_cols=['Chr', target])

    method = method.lower()
    if not method in ['scale', 'standardize', 'smooth', 'pad']:
        raise ValueError('`method` must be one of \'scale\', \'standardize\', or \'smooth\'.')

    # identify chromosomes.
    chroms = df['Chr'].unique()

    # Group by chromosome and scale target to fall between 0 and 1.
    tmp = df[['Chr', target]].copy()
    g = tmp.groupby('Chr')

    new_target_name = target + '_' + method + 'd'
    if method == 'scale':
        tmp = g.apply(lambda x: (x[target] - np.nanmin(x[target])) / (np.nanmax(x[target] - np.nanmin(x[target]))))

    if method == 'standardize':
        tmp = g.apply(lambda x: (x[target] - np.nanmean(x[target])) / (np.nanstd(x[target])))

    if method == 'smooth':
        tmp = g.apply(lambda x: DataFrame(smooth_signal(x[target].values
                                                        , smooth_window_len=kwargs.get('smooth_window_len', 11)
                                                        , smooth_window=kwargs.get('smooth_window', 'hamming')))[0])
        new_target_name = target + '_smoothed'

    if method == 'pad':
        tmp = g.apply(lambda x: DataFrame(pad_binary_signal(x[target].values
                                                            , pad_len=kwargs.get('pad_len', 5)))[0])
        new_target_name = target + '_padded'

    if method == 'proximity':
        tmp = g.apply(lambda x: DataFrame(pyramid_proximity_convolution(x[target].values)))
        new_target_name = target + '_proximity'

    df[new_target_name] = 0.0
    for chrom in chroms:
        chrom_idx = np.where(df['Chr'] == chrom)[0]
        df.iloc[chrom_idx, df.columns.get_loc(new_target_name)] = tmp.loc[chrom].values

    return df

