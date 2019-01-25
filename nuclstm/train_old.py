import itertools
from operator import itemgetter
from functools import reduce
from preprocessors import *


def gen_seq_scans(seq_df,
                  target='NCP/noise',
                  lookback=500,
                  shuffle=True,
                  shuffle_between=False,
                  batch_size=128,
                  step_size=1,
                  target_position='median',
                  no_one_class_samples=False,
                  flatten_final_target_dim=False,
                  testing=False):
    """
    Get a data generator that returns a batch (sample array, target array)
    of embbeded nucleotide sequences from a pandas.DataFrame. Each sample is a (lookback x 4)
    one-hot encoding of a DNA subsequence. Use to train an RNN in Keras.

    Processes/transforms data via the following:
    1. Identifies consecutive blocks of non-missing data with at least lookback-many points.
    2. Select batch_size-many starting positions to look backwards lookback-many positions
    for each individual DNA subsequence selection.
    3. One-hot-encodes each of the batch-many selected DNA subsequences.
    4. Selects corresponding target values. What target values per scan are dictated by
    the `target_position` argument.

    :param seq_df: pandas.DataFrame with a `seq` and a target column
    :param target: str name or int column index of the target data within `seq_df`
    :param lookback: int, size of DNA subsequences used to train an RNN
    :param shuffle: boolean should batches be shuffled internally? If False,
    scans will be selected in order, otherwise contiguous scans will be selected independently of each other.
    :param shuffle_between: boolean should batches be selected contiguously, or can each batch be
    selected independently of the prior batch?
    :param batch_size: int, number of DNA subsequence scans per generated tuple
    :param step_size: int, if shuffle is True, number of consecutive indices to
     step between samples in a batch.
    :param target_position: int or str, if str must be one of 'median' or 'all':
        - 'median' means "use target value in the median DNA subsequence position"
        - 'all' means the entire subsequence of target values should be used (e.g. for seq2seq learning)
        - int value means to use the target_position'th position of the lookback window.
    :param flatten_final_target_dim: Boolean should the `targets` have their final dimension flattened?
    E.g. LSTMs need 3-d input, but Dense final layers need 2-d input.
    :param testing: Boolean setting switch. If testing this function, generator yields a 3rd
    element of the tuple - the indices selected for the samples of the batch.

    :return: (samples, targets) tuple generator.
    `samples` will be a 3-d numpy.ndarray of shape [batch_size, lookback, 4].
    `targets` will be either a 2-d numpy.ndarray or a 1-d based on whether target_position is `all` or not,
    in the case of seq2seq learning
    """

    # data quality checks regarding seq_df DataFrame parameter
    check_dataframe_validity(seq_df
                             , reqd_cols=['seq', target])

    # ---------------------------------------------------------------------------- #
    # select function for scanning target vector during batch creation.
    # ---------------------------------------------------------------------------- #
    sequence_targets = False

    # select median position of a batch sequence.
    if target_position == 'median':
        target_idx = int(np.ceil(np.median(range(lookback))))
        target_len = 1

        def target_processor(vec, sub_idx):
            return np.mean(vec[sub_idx][target_idx]), sub_idx[target_idx]

    # select all positions of a batch sequence.
    elif target_position == 'all':
        sequence_targets = True
        target_len = lookback

        def target_processor(vec, sub_idx):
            return vec[sub_idx], sub_idx

    # select predetermined single or multiple positions of a batch sequence,
    # e.g. positions [center - 100: center + 100]
    elif isinstance(target_position, int) or isinstance(target_position, list):
        if len(target_position) > 1:
            sequence_targets = True
            target_len = len(target_position)

        def target_processor(vec, sub_idx):
            return vec[sub_idx][target_position], sub_idx[target_position]

    else:
        raise ValueError('`target_position` {0} not understood.'.format(target_position))

    # identify chromosomes in the set.
    chromosomes = seq_df['Chr'].unique().tolist()
    chromosomes.sort()

    # identify blocks of nonmissing data, per chromosome so that
    # consecutive row indices do not bleed over chromosomes.
    nonmissing_blocks = []
    for chrom in chromosomes:
        nonmissing_idx = np.where((seq_df[['seq', target]].notna()
                                   .all(axis=1)) & (seq_df['Chr'] == chrom))[0].tolist()
        for k, g in itertools.groupby(enumerate(nonmissing_idx), lambda ix: ix[0] - ix[1]):
            nonmissing_blocks.append(list(map(itemgetter(1), g)))

    # get rid of blocks that are shorter than scan window, and trim blocks down
    # to valid scan start positions only.
    nonmissing_blocks = [block[(lookback - 1):] for block in nonmissing_blocks if len(block) >= lookback]

    # collapse nonmissing_blocks list of lists to just one list of valid scan start positions and sort it.
    scan_idx = reduce(lambda x, y: x + y, nonmissing_blocks)
    scan_idx.sort()
    n_scans = len(scan_idx)
    if n_scans == 0:
        raise ValueError('No length-{0} blocks of contiguous sequence data found! Cannot scan.'.format(lookback))

    # set values we'll need during batch generation.
    x = seq_df['seq'].values
    y = seq_df[target].values
    i = 0

    while True:

        # if shuffling between batches, reset the batch starting index, i.
        if shuffle_between:
            i = np.random.choice(range(n_scans)
                                 , size=1)[0]

        # if shuffling within a batch, select batch_size-many end-of-scan indices.
        if shuffle:
            scan_ends = np.random.choice(scan_idx
                                         , size=batch_size
                                         , replace=False)

        # if not shuffling within batches, select batch_size-many end-of-scan indices in order,
        # abiding by step-size gaps.
        else:
            scan_ends = []
            for _ in range(batch_size):

                if i >= n_scans:
                    i = i % n_scans

                scan_ends.append(scan_idx[i])

                # within a batch, move to the next subsequence end position.
                i += step_size

        # batch samples storage 3-d np.ndarray.
        samples = np.zeros([len(scan_ends), lookback, 4])

        # if doing seq2seq learning, targets have to be full sequences (2-d array), otherwise
        # targets will just be a 1-d array of len(scan_ends), which is typically just batch_size.
        if sequence_targets:
            targets = np.zeros([len(scan_ends), target_len, 1])
        else:
            targets = np.zeros([len(scan_ends), ])

        if testing:
            idx = list()

        # load up batch's samples and targets.
        for ii in range(len(scan_ends)):
            eos = int(scan_ends[ii])

            targets_tmp, selected_idx = target_processor(y
                                                         , sub_idx=np.arange((eos + 1 - lookback), (eos + 1)))

            # controlling if targets are sequences different than if they're scalars.
            if sequence_targets:

                # if user does not want samples where target sequences are all one value (e.g. negative class),
                # try 1000 times to resample positions until a target sequence with > 1 classes found.
                if no_one_class_samples:
                    stddev = np.std(targets_tmp)
                    max_tries = 1000
                    try_ctr = 1
                    while stddev == 0.0 and try_ctr <= max_tries:
                        if shuffle:
                            eos = int(np.random.choice(scan_idx
                                                       , size=1))
                        else:
                            raise NotImplementedError(
                                'Sequential batch creation zero-variance avoidance not implemented.')

                        targets_tmp, selected_idx = target_processor(y
                                                                     , sub_idx=np.arange((eos + 1 - lookback), (eos + 1)))
                        try_ctr += 1
                        stddev = np.std(targets_tmp)

                targets[ii] = np.expand_dims(targets_tmp, axis=1)
            else:
                targets[ii] = targets_tmp

            samples[ii] = embed_nucleotide_seq(x[(eos + 1 - lookback):(eos + 1)])

            # if testing is specified, identify the indices of the dataframe used in batch's samples.
            if testing:
                idx.extend(selected_idx)

        # flatten final target dimension if desired, e.g. Dense layers
        # will only accept 2-d input, so sequence targets need to be 1-dim, so targets.shape = (x, y).
        if flatten_final_target_dim:
            targets = targets.reshape(-1, targets.shape[1])

        samples = samples.astype('float32')
        targets = targets.astype('float32')

        if np.isnan(samples).any():
            raise ValueError('NAN values found in np.ndarray `samples` for scan_ends = {0}'.format(scan_ends))
        if np.isnan(targets).any():
            raise ValueError('NAN values found in np.ndarray `targets` for scan_ends = {0}'.format(scan_ends))

        if testing:
            yield samples, targets, idx

        else:
            yield samples, targets


if __name__ == '__main__':

    test_df = DataFrame({'seq': [l for l in 'ATTACCCGGGAGGATA']
                            , 'Chr': 'chrI'
                            , 'pos': list(range(1, 17))
                            , 'nucleosome': [0] * 10 + [1] + [0] * 2 + [1] + [0] * 2})

    gen = gen_seq_scans(test_df
                        , target='nucleosome'
                        , lookback=10
                        , batch_size=2
                        , shuffle=False
                        , shuffle_between=True
                        , target_position='all'
                        , testing=True)
    tmp_x, tmp_y, idx = next(gen)

    print(tmp_x[0:2])
    print(tmp_y[0:2])
    print(idx[0:2])