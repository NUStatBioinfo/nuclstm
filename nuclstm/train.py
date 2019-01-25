from preprocessors import *


def gen_seq_scans(seq_df,
                  target='nucleosome',
                  lookback=250,
                  shuffle=True,
                  consecutive_batches=False,
                  batch_size=128,
                  step_size=1,
                  target_position='median',
                  no_one_class_samples=False,
                  flatten_final_target_dim=False,
                  return_idx=False):
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
    :param shuffle: boolean should samples within batches be shuffled? If False,
    scans will be selected in order, otherwise contiguous scans will be selected independently of each other.
    :param consecutive_batches: Boolean; should there be shuffling between batches?
    Recommended this is set to False for RNN type models.
    :param batch_size: int, number of DNA subsequence scans per generated tuple
    :param step_size: int, if shuffle is True, number of consecutive indices to
     step between samples in a batch.
    :param target_position: int or str, if str must be one of 'median' or 'all':
        - 'median' means "use target value in the median DNA subsequence position"
        - 'all' means the entire subsequence of target values should be used (e.g. for seq2seq learning)
        - int value means to use the target_position'th position of the lookback window.
    :param flatten_final_target_dim: Boolean should the `targets` have their final dimension flattened?
    E.g. LSTMs need 3-d input, but Dense final layers need 2-d input.
    :param return_idx: boolean; if indices of input df are desired, generator will yield a 3rd
    element of the tuple - the indices selected for the samples of each batch.

    :return: (samples, targets) tuple generator.
    `samples` will be a 3-d numpy.ndarray of shape [batch_size, lookback, 4].
    `targets` will be either a 2-d numpy.ndarray or a 1-d based on whether target_position is `all` or not,
    in the case of seq2seq learning
    """

    # subset to data we actually need
    seq_df = seq_df[['Chr', 'seq', 'pos', target]]
    seq_df = seq_df[~seq_df.isnull().any(1)]

    if seq_df.empty:
        raise ValueError('sequence data is empty after null removal!')

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

    # identify blocks of nonmissing data, per chromosome so that
    # consecutive row indices do not bleed over chromosomes.
    # assume unique_full_df has had NaNs removed.
    chromosomes = seq_df['Chr'].unique().tolist()
    chromosomes.sort()
    cts_regions = list()

    # for each chromosome, build up list of [start, end] indices of continuous position runs
    for chrom in chromosomes:

        # pick out indices of DataFrame for this chromosome.
        chrom_idx = np.where(seq_df.Chr == chrom)[0]
        chrom_offset = np.min(chrom_idx)
        chrom_len = len(chrom_idx)

        sub_df = seq_df.iloc[chrom_idx][['seq', 'pos']]
        pos_vec = sub_df.pos.values

        # determine where consecutive change change in pos is > 1
        changepoints = np.where(np.diff(pos_vec) > 1)[0]
        n_changepoints = len(changepoints)
        n_regions = n_changepoints + 1

        # special case for when whole chromosome is one contiguous region,
        # [start, end] = [chromosome start index, chromosome end index].
        if n_regions == 1:
            cts_regions.append([chrom_offset, chrom_offset + chrom_len - 1])

        # construct [start, end] contiguous region indices.
        else:
            for i in range(n_regions):
                if i == 0:
                    start = chrom_offset
                    end = chrom_offset + changepoints[i]
                elif i < n_regions - 1:
                    start = chrom_offset + changepoints[i - 1] + 1
                    end = chrom_offset + changepoints[i]
                else:
                    start = chrom_offset + changepoints[i - 1] + 1
                    end = chrom_offset + chrom_len - 1

                cts_regions.append([start, end])

    # filter out continuous sub-regions that aren't long enough to build batch specifications.
    if (not shuffle) or (consecutive_batches):
        scan_regions = [x for x in cts_regions if x[1] - x[0] >= lookback * batch_size]

    else:
        scan_regions = [x for x in cts_regions if x[1] - x[0] >= lookback]

    # ensure we have data to train with.
    if len(scan_regions) == 0:
        raise ValueError('No region sufficiently long for batch creation. '
                         'Check that lookback or lookback*batch_size is reasonable')

    # determine region lengths.
    region_lens = [x[1] - x[0] for x in scan_regions]
    region_lens /= np.sum(region_lens)
    n_regions = len(scan_regions)

    # extract raw, unselected training and target data
    x = seq_df['seq'].values
    y = seq_df[target].values

    # if generating batches for test set evaluation, initialize generator to start at first viable
    # position in first continuous region.
    if consecutive_batches:
        region_idx = 0
        region = scan_regions[region_idx]
        end_pos = region[0] + lookback - 1

    while True:

        # storage for indices marking ends of samples.
        scan_ends = np.zeros(batch_size
                             , dtype=int)

        if consecutive_batches:
            # if test set batch generation sample index falls outside of range of contiguous region,
            # then move to next contiguous region, modulo the number of continuous regions there are.
            if end_pos > region[1] - lookback * (batch_size - 1):
                region_idx = 0 if (region_idx + 1) % n_regions == 0 else region_idx + 1
                region = scan_regions[region_idx]
                end_pos = region[0] + lookback - 1

            # Now add sample end indices to the batch.
            for i in range(batch_size):
                scan_ends[i] = end_pos
                end_pos += lookback  # increment samples in test set batch by size of the samples.

        else:
            # if not shuffling within a batch, randomly select a continuous region to scan,
            # and then pick a continuous region of lookback * batch_size scan indices.
            if not shuffle:
                # randomly select one continuous region with probability according to length of cts region.
                region = scan_regions[np.random.choice(range(n_regions)
                                                       , size=1
                                                       , p=region_lens)[0]]

                # assemble set of possible sample-ends indices.
                # sample-ends designate the end of a sample that can start a batch.
                scan_idx = range(region[0] + lookback - 1, region[1] - lookback * (batch_size - 1) + 1)

                # randomly select a starting index for batch creation.
                end_pos = np.random.choice(scan_idx
                                           , size=1)[0]

                for i in range(batch_size):
                    # if index selection is too large because of step_size, reset to earlier in the cts region.
                    if end_pos > region[1]:
                        end_pos = region[0] + lookback - 1

                    # append sample end index from region.
                    scan_ends[i] = end_pos

                    # within a batch, move to the next subsequence end position.
                    end_pos += step_size

            # if shuffling within a batch, randomly select legal continuous regions.
            else:
                # randomly select batch_size-many continuous regions, one sample within each selection.
                for i in range(batch_size):
                    # pick region.
                    region = scan_regions[np.random.choice(range(n_regions)
                                                           , size=1
                                                           , p=region_lens)[0]]

                    # pick sample end index from region.
                    scan_ends[i] = np.random.choice(range(region[0] + lookback - 1, region[1] + 1)
                                                    , size=1
                                                    , replace=False)[0]

        # output storage
        samples = np.zeros([batch_size, lookback, 4])

        # if doing seq2seq learning, targets have to be full sequences (2-d array), otherwise
        # targets will just be a 1-d array of len(scan_ends), which is typically just batch_size.
        if sequence_targets:
            targets = np.zeros([len(scan_ends), target_len, 1])
        else:
            targets = np.zeros([len(scan_ends), ])

        if return_idx:
            idx = list()

        # load up batch's samples and targets.
        for ii in range(len(scan_ends)):
            eos = int(scan_ends[ii])

            targets_tmp, selected_idx = target_processor(y
                                                         , sub_idx=np.arange((eos - lookback + 1), eos + 1))

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
                            eos = int(np.random.choice(scan_ends
                                                       , size=1))
                        else:
                            raise NotImplementedError(
                                'Sequential batch creation zero-variance avoidance not implemented.')

                        targets_tmp, selected_idx = target_processor(y
                                                                     , sub_idx=np.arange((eos - lookback + 1), eos + 1))
                        try_ctr += 1
                        stddev = np.std(targets_tmp)

                targets[ii] = np.expand_dims(targets_tmp, axis=1)
            else:
                targets[ii] = targets_tmp

            samples[ii] = embed_nucleotide_seq(x[(eos - lookback + 1): (eos + 1)])

            # if return_idx is specified, identify the indices of the dataframe used in batch's samples.
            if return_idx:
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

        if return_idx:
            yield samples, targets, idx

        else:
            yield samples, targets


if __name__ == '__main__':
    from pandas import read_feather

    df = read_feather('/Users/fineiskid/nu/jiping_research/data/unique_nucleosome_map.feather')

    gen = gen_seq_scans(df
                        , target='nucleosome'
                        , lookback=250
                        , batch_size=128
                        , shuffle=False
                        , target_position='all'
                        , consecutive_batches=True
                        , return_idx=True)
    tmp_x, tmp_y, idx = next(gen)

    print('first bit of sequence: {0}'.format(df.iloc[idx[0:10]]['seq'].values))
    print('tmp_x.shape = {0}'.format(tmp_x.shape))
    print('tmp_y.shape = {0}'.format(tmp_y.shape))
    print('len(idx) = {0}'.format(len(idx)))

    print(tmp_x[0:2])
    print(tmp_y[0:2])
    print(idx[0:2])