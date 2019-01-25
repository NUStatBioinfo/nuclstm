import numpy as np
import logging
import os
from platform import platform
from re import search
from pandas import DataFrame
import matplotlib

if search('Linux', platform()):
    matplotlib.use('agg')

import pylab as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from utils import check_dataframe_validity
from preprocessors import pad_binary_signal


def extract_preds_from_test_set(gen, model, reset_data=None):
    """
    Run a model on specified number of batches extracted from a generator object.

    :param gen: generator object that returns a batch of (test set samples, test set targets).
    Generator should come from train.gen_seq_scans(..., test_set=True) so that batches are created sequentially.
    :param model: keras.models.Sequential model object
    :param reset_data: pandas.DataFrame with data from which generator came from, use to determine
    when to reset model states. Use only with a stateful model.
    :return: y, preds 2-tuple of arrays, the original target array (y) and the
    model predicted array (preds)
    """
    y = list()
    preds = list()
    idx = list()

    proceed = True
    batch_ctr = 0
    prev_end_idx = None

    while proceed:
        if (batch_ctr % 10 == 0) and (batch_ctr > 0):
            print('batch number {0}, total predictions made = {1}'.format(batch_ctr, len(y)))

        # if reset_every is an integer, reset model every n batches.
        x_batch, y_batch, idx_batch = next(gen)

        # if using a stateful model and there's a discontinuity between consecutive batches, reset model.
        if (reset_data is not None) and (batch_ctr > 0):
            batch_step = reset_data.pos.iloc[idx_batch[0]] - reset_data.pos.iloc[prev_end_idx]
            prev_end_idx = idx_batch[-1]

            if batch_step != 1:
                print('batch discontinuity found. resetting model state.')
                model.reset_states()

        # store end index from current batch.
        else:
            prev_end_idx = idx_batch[-1]

        preds_batch = model.predict(x_batch
                                    , batch_size=x_batch.shape[0])

        y += y_batch.ravel().tolist()
        preds += preds_batch.ravel().tolist()
        idx += idx_batch

        # check whether or not any batch indices are repeated. If they are,
        # this means that batches have reset to the beginning of the test set
        # as the entire test set has been covered.
        proceed = len(set(idx)) == len(idx)
        batch_ctr += 1

    # remove duplicate predictions and arrange predictions in genome order.
    preds_df = DataFrame({'y': y
                          , 'preds': preds
                          , 'idx': idx})
    preds_df.drop_duplicates(subset=['idx']
                             , inplace=True)
    preds_df.sort_values(['idx']
                         , inplace=True)
    preds_df.reset_index(drop=True
                         , inplace=True)

    return preds_df


def pr_plot(y, preds, fname=None, figsize=(8, 6), verbose=True):
    """
    Plot a precision-recall curve and compute the AUC.

    :param y: numpy 1-d array of true target data
    :param preds: numpy 1-d array of target predictions
    :param fname: filename of saved plot, default is None (do not save)
    :param figsize: (width in, height in) tuple
    :param verbose: boolean, if True write precision-recall plot AUC to logger
    :return: None
    """
    precision, recall, thresh = precision_recall_curve(y
                                                       , probas_pred=preds)
    pr_auc = auc(recall, precision)

    fig = plt.figure(figsize=figsize)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-recall curve ---- AUC = {:6.4f}'.format(pr_auc))

    if fname:
        plt.savefig(fname)

    if verbose:
        logging.info('Precision-recall AUC = {:6.4f}'.format(pr_auc))


def roc_plot(y, preds, fname=None, figsize=(8, 6), verbose=True):
    """
    Plot true positive rate vs false positive rate (ROC) curve and compute the AUC.

    :param y: vector of true target data
    :param preds: vector of target predictions
    :param fname: filename of saved plot, default is None (do not save)
    :param figsize: (width in, height in) tuple
    :param verbose: boolean, if True write ROC plot AUC to logger
    :return: None
    """
    fpr, tpr, thresholds = roc_curve(y
                                     , y_score=preds)
    roc_auc = roc_auc_score(y
                            , y_score=preds)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve: AUC = {0}'.format(roc_auc))

    if fname:
        plt.savefig(fname)

    if verbose:
        logging.info('ROC AUC = {:6.4f}'.format(roc_auc))


def sensitivity_kdist_plot(df, fname,
                           preds1='nupop_ncp', preds2='nuclstm_ncp',
                           figsize=(8, 6)):
    """
    Compare the sensitivities of two nucleosome center prediction methods
    by calling a "successful" location when a predicted ncp is
     +/- k-base pairs distance of some ground truth 'nucleosome' vector.

     See Figure 3 in Wang, Widom (2010).

    :param df: pandas.DataFrame with a 'nucleosome' field along with preds1 and preds2 fields.
    :param fname: str filename to save plot
    :param preds1: str name of first model's predicted nucleosome center positions
    :param preds2: str name of second model's predicted nucleosome center positions
    :param figsize: (width in, height in) tuple
    """
    reqd_cols = [preds1, preds2, 'nucleosome']
    check = check_dataframe_validity(df
                                     , reqd_cols=reqd_cols)

    sub_df = df[reqd_cols]
    n_nucs = sub_df['nucleosome'].sum()

    pad_lens = list(np.arange(5, stop=75, step=5))
    pad_lens.append(73)
    nupop_tpr = []
    nuclstm_tpr = []

    for pad_len in pad_lens:
        sub_df.loc[:, 'nucleosome_padded'] = pad_binary_signal(sub_df['nucleosome']
                                                               , pad_len=pad_len)

        tpr = ((sub_df[preds1] == 1) & (sub_df['nucleosome_padded'] == 1)).sum() / n_nucs
        nupop_tpr.append(100 * tpr)

        tpr = ((sub_df[preds2] == 1) & (sub_df['nucleosome_padded'] == 1)).sum() / n_nucs
        nuclstm_tpr.append(100 * tpr)

    plt.figure(figsize=figsize)
    plt.plot(pad_lens, nupop_tpr, 'b--', label=preds1)
    plt.plot(pad_lens, nuclstm_tpr, 'r-.', label=preds2)
    plt.legend()
    plt.ylabel('sensitivity (%)')
    plt.xlabel('distance (bp)')
    plt.title('Sensitivity vs. padding distance from ground truth')
    plt.savefig(fname)


def plot_training_history(history, fname=None,
                          metric_name='acc', figsize=(8, 6)):
    """
    Plot the history of a model's loss metrics during training.

    :param history: dict with 'loss', `metric_name` elements. Optionally,
    contains 'val_epochs', 'val_loss', and 'val_'+`metric_name` elements to plot validation performance.
    :param fname: str filename to save plot
    :param metric_name: str name of extra metric monitored during training
    :param figsize: (width in, height in) tuple
    :return: None
    """
    epochs = np.arange(1, len(history['loss']) + 1)

    val = 'val_loss' in history and 'val_epochs' in history

    if metric_name:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(epochs, history['loss'], 'b-')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_title('Model training: loss')

        axes[1].plot(epochs, history[metric_name], 'b-', label=metric_name)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title('Model training: {0}'.format(metric_name))

        if val:
            axes[0].plot(history['val_epochs'], history['val_loss'], 'r-', label='Validation loss')
            axes[1].plot(history['val_epochs'], history['val_' + metric_name], 'r-', label='Validation ' + metric_name)

        axes[1].legend()

    else:
        fig = plt.figure(figsize=figsize)
        plt.plot(epochs, history['loss'], 'b-', label='Loss')
        plt.title('Model training: loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if val:
            plt.plot(epochs, history['val_loss'], 'r-', label='Validation loss')

        plt.legend()

    if fname:
        plt.savefig(fname)


def plot_genome_predictions(y, preds, fname=None, width=2000, figsize=(14, 14)):
    """
    Plot 2x2 panel of 4 subplots of predicted vs. actual nucleosome center position
    model output for a particular chromosome.

    :param y: np.array vector of ground truth
    :param preds: np.array vector of predictions
    :param fname: str filename to save plot
    :param width: int width of subplots in terms of base positions
    :param figsize: (width, height) tuple
    :return: None
    """
    if len(y) != len(preds):
        raise ValueError('len(y) not equal to len(preds)!')

    # break up chromosome into chunks for plotting.
    starts = np.linspace(0, stop=len(y), num=6, dtype=int)[1:5]

    # if plot width actually longer than len(chromosome) / 6, shrink
    # the width down to len(chromosome) / 6.
    if width > starts[1] - starts[0]:
        width = starts[1] - starts[0]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    idx = 0
    for i in range(2):
        for j in range(2):
            x1 = starts[idx]
            x2 = x1 + width

            x = np.arange(x1, x2)
            axes[i, j].plot(x, y[x1:x2], 'b-')
            axes[i, j].plot(x, preds[x1:x2], 'r-')
            axes[i, j].set_ylabel('Nucleosome occupancy')
            axes[i, j].set_xlabel('base position')
            axes[i, j].set_title('Test set evaluation bases {0}:{1}'.format(x1, x2))
            idx += 1

    if fname:
        plt.savefig(fname)


def plot_correlation_heatmap(df, fname=None, figsize=(10, 10)):
    """
    Make a correlation matrix heatmap from a pandas.DataFrame using the
    Seaborn library.

    :param df: pandas.DataFrame from which correlation matrix will be computed
    :param fname: str filename to save plot
    :param figsize: (width, height) tuple. Square is recommended.
    :return: None
    """
    corr = df.corr()
    fig = plt.figure(figsize=figsize)
    sns.heatmap(corr
                , xticklabels=corr.columns.values
                , yticklabels=corr.columns.values
                , cmap='bwr'
                , annot=True)

    if fname:
        fig.savefig(fname
                    , bbox_inches='tight')


def plot_cross_correlation(y, preds, fname,
                           lag_width=80, figsize=(8, 6)):
    """
    Plot two vectors' cross-correlation function. Cross-correlation
    is defined as (f*g)(j) = sum_{i=-\infty}^{\infty}f(i)g(i+j)

    :param y: np.array of ground truth values
    :param preds: np.array of predicted values
    :param fname: str filename to save plot
    :param lag_width: int value defining range of cross-correlation function to compute/plot;
     defined as [-|lag_width|, +|lag_width|]
    :param figsize: (width in, height in) tuple
    """
    lag_width = np.abs(lag_width)

    if len(y) != len(preds):
        raise ValueError('len(y) != len(preds)')

    lag_vec = np.arange(-lag_width
                        , lag_width + 1)
    # cc_vec = np.copy(lag_vec)
    #
    # for i in range(len(lag_vec)):
    #     lag = lag_vec[i]
    #     (stag_vec, lagged_vec) = (preds, y)
    #
    #     if lag > 0:
    #         (stag_vec, lagged_vec) = (y, preds)
    #
    #     if lag != 0:
    #         cc_vec[i] = np.dot(stag_vec[np.abs(lag):]
    #                            , b=lagged_vec[:(-np.abs(lag))])
    #
    #     else:
    #         cc_vec[i] = np.dot(stag_vec
    #                            , b=lagged_vec)

    # use numpy.convolve with reversed "v" vector. See https://bit.ly/2EYlFpd
    cc_vec = np.convolve(y
                         , v=preds[::-1]
                         , mode='full')
    mid_pos = np.median(range(len(cc_vec)))
    cc_idx = np.arange(mid_pos - lag_width
                       , stop=mid_pos + lag_width + 1
                       , dtype=int)

    plt.figure(figsize=figsize)
    plt.plot(lag_vec, cc_vec[cc_idx], '-g', linewidth=1.2)
    plt.ylabel('Cross-correlation')
    plt.xlabel('Lag')
    plt.title('Nucleosome map/predicted centers cross-correlation function')
    plt.savefig(fname)


def evaluate_binary_output_model(preds_df, chrom_name, save_dir=None, verbose=True):
    """
    Evaluate the performance of a binary classifier model on a chromosome.

    :param preds_df: pandas.DataFrame with test set target values and predictions. See extract_preds_from_test_set.
    :param chrom_name: name of chromosome we're evaluating model on
    :param save_dir: path to directory where output figures will be saved
    :param verbose: boolean, if True write metrics to logger
    :return: None
    """
    pr_plot(preds_df.y.values
            , preds=preds_df.preds.values
            , fname=os.path.join(save_dir, 'precision_recall_' + chrom_name + '.png') if save_dir else None
            , verbose=verbose)

    roc_plot(preds_df.y.values
             , preds=preds_df.preds.values
             , fname=os.path.join(save_dir, 'roc_' + chrom_name + '.png') if save_dir else None
             , verbose=verbose)

    plot_genome_predictions(preds_df.y.values
                            , preds=preds_df.preds.values
                            , fname=os.path.join(save_dir, 'pred_plot_' + chrom_name + '.png') if save_dir else None
                            , width=2000
                            , figsize=(14, 14))

    preds_bin = np.zeros(preds_df.shape[0])
    preds_bin[preds_df.preds >= 0.5] = 1.0
    est_acc = np.mean(preds_df.y == preds_bin)

    if verbose:
        logging.info('accuracy using @ probability threshold of 0.5 = {:6.4f}'
                     .format(est_acc))


def evaluate_cts_output_model(preds_df, chrom_name, save_dir=None,
                              threshold=0.05, verbose=True):
    """
    Evaluate the performance of a regression model whose output is a continuous value.

    :param preds_df: pandas.DataFrame with test set target values and predictions. See extract_preds_from_test_set.
    :param chrom_name: name of chromosome we're evaluating model on
    :param save_dir: path to directory where output figures will be saved
    :param threshold: float threshold above which we consider target/prediction values to be 1's,
    and lower than that, 0's
    :param verbose: boolean, if True write metrics to logger
    :return: None
    """
    plot_genome_predictions(preds_df.y.values
                            , preds=preds_df.preds.values
                            , fname=os.path.join(save_dir, 'pred_plot' + chrom_name + '.png') if save_dir else None
                            , width=2000
                            , figsize=(14, 14))

    corr = None
    true_acc_vec = np.zeros(preds_df.shape[0])
    pred_acc_vec = np.zeros(preds_df.shape[0])
    true_acc_vec[preds_df.y >= threshold] = 1.0
    pred_acc_vec[preds_df.preds >= threshold] = 1.0
    acc = np.mean(true_acc_vec == pred_acc_vec)

    if preds_df.y.std() != 0.0 and preds_df.preds.std() != 0.0:
        corr = np.corrcoef(preds_df.y, preds_df.preds)[0, 1]

        pr_plot(true_acc_vec
                , preds=preds_df.preds.values
                , fname=os.path.join(save_dir, 'precision_recall' + chrom_name + '.png') if save_dir else None
                , verbose=verbose)

        roc_plot(true_acc_vec
                 , preds=preds_df.preds.values
                 , fname=os.path.join(save_dir, 'roc' + chrom_name + '.png') if save_dir else None
                 , verbose=verbose)

    if corr and verbose:
        logging.info('correlation(true, predicted sequences) = {:6.4f}'
                     .format(corr))

    if verbose:
        logging.info('binary accuracy @ threshold = {0}: {:6.4f}'
                     .format(threshold, acc))