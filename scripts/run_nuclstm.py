#!/bin/python

# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
# Run script to train and evaluate a neural network that locates nucleosome center positions.
# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
import sys
import os
import logging
import pickle as pkl
import shutil
from keras.callbacks import CSVLogger, EarlyStopping, TerminateOnNaN, ModelCheckpoint
from keras.models import load_model
from keras import backend as K
import pprint
sys.path.append(os.getenv('NUCLSTM_SOURCE'))
from math import inf
from loaders import *
from train import *
from utils import *
from models import *
from preprocessors import *
from evaluation import *
from model_helpers import *
from postprocessors import *


# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
if __name__ == '__main__':

    # ---------------------------------------------------------------------- #
    # 1. Load config file, configure logger, parse run parameters
    # ---------------------------------------------------------------------- #

    # parse command line arguments, read in config file.
    args = parse_nuc_args()
    params = ConfigLoader(args.config_file).load()

    MODEL_NAME = params['run_params']['model_name']
    MODEL_TYPE = params['run_params']['model_type'].lower()
    BINARY = params['run_params']['binary']
    TARGET = params['model_params']['target']

    check = validate_model_type(MODEL_TYPE)

    # create output directory (if specified) and set up logger.
    output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # copy config file to output dir.
    shutil.copy(args.config_file
                , os.path.join(output_dir, os.path.basename(args.config_file)))

    # set up logger.
    handlers = [logging.FileHandler(os.path.join(output_dir, MODEL_NAME + '.txt'))
                , logging.StreamHandler()]
    logging.basicConfig(handlers=handlers
                        , level=logging.DEBUG
                        , format='NUCLSTM (%(asctime)s) ---- %(message)s'
                        , datefmt='%m/%d/%Y %I:%M:%S')

    # set float type.
    if params['run_params']['floatx']:
        fltx = 'float' + str(params['run_params']['floatx'])
        logging.info('Session float type: {0}'.format(fltx))
        K.set_floatx(fltx)

    # print out args and model config/settings.
    logging.info('Cmd line arguments: {0}'.format(pprint.pformat(args)))
    logging.info('Params: {0}'.format(pprint.pformat(params)))

    # break up chromosome strings into lists of integers.
    all_chroms = args.train_chromosomes + args.val_chromosomes + args.test_chromosomes

    # identify columns for which we want to plot correlation matrix heatmap.
    test_corr_cols = ['NCP', 'NCP/noise', 'NCP/noise_smoothed', 'nuclstm_preds']

    # determine if training a stateful RNN.
    try:
        stateful = params['model_params']['rnn_params']['stateful']
    except KeyError:
        stateful = False

    # ---------------------------------------------------------------------- #
    # 2. Load in nucleosome positioning data, preprocess data
    # ---------------------------------------------------------------------- #

    # load in nucleosome positioning data.
    logging.info('Loading nucleosome positioning data from {0}.'.format(args.input_file))
    loader = NcpLoader(os.path.dirname(args.input_file))
    nuc_df = loader.load(os.path.basename(args.input_file)
                         , chromosomes=all_chroms)
    if TARGET not in nuc_df.columns:
        raise ValueError('Target column {0} is not in loaded ncp dataset!'.format(TARGET))

    logging.info('Successfully loaded nucleosome positioning data. Shape = {0}.'.format(nuc_df.shape))

    # preprocess the target vector if specified, e.g. for continuous transformations of the nucleosome binary vector.
    if params['preprocessor_params'].get('target', None):
        logging.info('Preprocessing target vector {0} via {1} method.'
                     .format(TARGET, params['preprocessor_params']['target']['method']))
        orig_cols = nuc_df.columns.tolist()
        nuc_df = preprocess_by_chromosome(nuc_df
                                          , target=TARGET
                                          , method=params['preprocessor_params']['target']['method']
                                          , smooth_window_len=params['preprocessor_params']['target']['smooth_window_len']
                                          , smooth_window=params['preprocessor_params']['target']['smooth_window']
                                          , pad_len=params['preprocessor_params']['target']['pad_len'])
        TARGET = list(set(nuc_df.columns.tolist()) - set(orig_cols))[0]
        logging.info('Target vector successfully transformed into {0}'.format(TARGET.upper()))

    # log descriptive statistics of target variable.
    logging.info('Target {0} summary: {1}'.format(TARGET, nuc_df[TARGET].describe()))

    # if comparison with NuPoP is desired, load test chromosomes' NuPoP output.
    if args.nupop_dir:
        logging.info('Loading NuPoP output for test set chromosomes.')
        nupop_df_list = list()

        # stack nupop predictions per-chromosome into one DataFrame.
        for chrom in args.test_chromosomes:
            nupop_file = os.path.join(args.nupop_dir, 'chr_{0}_nupop.feather'.format(str(chrom)))
            nupop_df_list.append(loader.load_nupop_data(nupop_file))

        nupop_df = concat(nupop_df_list)
        logging.info('Successfully loaded NuPoP results for {0} chromosomes.'.format(len(args.test_chromosomes)))

    # ---------------------------------------------------------------------- #
    # 3. Construct data generators for model training.
    # ---------------------------------------------------------------------- #

    # create training/validation/test set data generators. Note: you cannot pickle up generator objects.
    logging.info('Creating data generators for keras model.fit_generator method. Target variable ---- {0}'
                 .format(TARGET))

    flatten = (MODEL_TYPE in ['dense', 'covrnn'] and params['data_params']['output_len'] > 1 and not stateful)

    if flatten:
        logging.info('Warning: `flatten` argument is True. y-batches will be 2-dimensional.')

    # assemble chromosome training, validation generators.
    train_df = loader.subset_to_chromosomes(nuc_df
                                            , chromosomes=args.train_chromosomes)
    train_gen = gen_seq_scans(train_df
                              , lookback=params['data_params']['lookback']
                              , batch_size=params['data_params']['batch_size']
                              , target=TARGET
                              , target_position=params['data_params']['target_position']
                              , shuffle=params['data_params']['shuffle']
                              , step_size=params['data_params']['step_size']
                              , flatten_final_target_dim=flatten
                              , no_one_class_samples=params['data_params']['no_one_class_samples']
                              , consecutive_batches=stateful
                              , return_idx=stateful)

    val_df = loader.subset_to_chromosomes(nuc_df
                                          , chromosomes=args.val_chromosomes)
    val_gen = gen_seq_scans(val_df
                            , lookback=params['data_params']['lookback']
                            , batch_size=params['data_params']['batch_size']
                            , target=TARGET
                            , target_position=params['data_params']['target_position']
                            , shuffle=params['data_params']['shuffle']
                            , step_size=params['data_params']['step_size']
                            , flatten_final_target_dim=flatten
                            , no_one_class_samples=params['data_params']['no_one_class_samples']
                            , consecutive_batches=stateful)

    val_x, val_y = next(val_gen)
    logging.info('x batch shape: {0}.'.format(val_x.shape))
    logging.info('y batch shape: {0}.'.format(val_y.shape))

    # data check.
    likely_binary = len(np.unique(val_y)) == 2
    if likely_binary and not BINARY:
        raise Warning('run_params.binary was set to false but only found 2 unique values in a sampled target vector.')

    # back out binary class weights if user specified auto class weighting.
    if params['model_params']['class_weight'] == 'auto':
        weights = get_binary_class_weights(loader.subset_to_chromosomes(nuc_df
                                                                        , args.train_chromosomes)[TARGET].values)
        logging.info('class_weights = {0}'.format(pprint.pformat(weights)))
        params['model_params']['class_weight'] = weights

    if not args.pretrained_model:

        # ---------------------------------------------------------------------- #
        # 4. Construct model, initialize loss storage
        # ---------------------------------------------------------------------- #

        # Construct model.
        logging.info('Building a {0} model with Keras.'.format(MODEL_TYPE))
        m = ModelBuilder(config=params).build()

        # Print model summary.
        m.summary(print_fn=lambda x: logging.info(x + '\n'))

        # initialize storage for best model.
        m_best = m

        # model checkpoint file.
        model_save_file = os.path.join(output_dir, 'model_{epoch:02d}-{loss:.5f}.h5')

        # Create vector of model checkpoint indices
        model_checkpoints = np.arange(1
                                      , stop=params['model_params']['epochs'] + 1
                                      , step=params['model_params']['checkpoint_every']
                                      , dtype=int)

        # Determine validation set evaluation checkpoints.
        val_checkpoints = np.arange(1
                                    , stop=params['model_params']['epochs'] + 1
                                    , step=params['model_params']['validate_every']
                                    , dtype=int)
        val_loss = []
        val_xtra_metric = []

        # Determine if model will leverage early stopping.
        es = params['model_params']['early_stopping']

        # ---------------------------------------------------------------------- #
        # 5. Train model, plot training diagnostics.
        # ---------------------------------------------------------------------- #

        # Begin RNN model training.
        if stateful:
            logging.info('Begin {0} model training (stateful = {1})...'.format(MODEL_TYPE, stateful))

            best_val_loss = inf
            best_val_loss_epoch = 0
            loss_history = []
            metrix_history = []
            batch_ctr = 0

            for epoch in range(1, params['model_params']['epochs'] + 1):
                tr_losses = []
                tr_metrix = []

                # Train model for one epoch.
                for step_idx in range(params['model_params']['steps_per_epoch']):
                    x_batch, y_batch, idx_batch = next(train_gen)

                    # reset states only when we encounter a discontinuity in training positions.
                    if batch_ctr > 0:
                        batch_step = train_df.pos.iloc[idx_batch[0]] - train_df.pos.iloc[prev_end_idx]
                        prev_end_idx = idx_batch[-1]

                        if batch_step != 1:
                            logging.info('Epoch {0} -- step {1} ---- batch discontinuity found. resetting model state.'
                                         .format(epoch, step_idx))
                            m.reset_states()

                    else:
                        prev_end_idx = idx_batch[-1]

                    if params['model_params']['class_weight']:
                        tr_loss, tr_metric = m.train_on_batch(x_batch
                                                              , y=y_batch
                                                              , class_weight=params['model_params']['class_weight'])
                    else:
                        tr_loss, tr_metric = m.train_on_batch(x_batch
                                                              , y=y_batch)

                    # if gradient blows up, reset state and progress on this epoch.
                    if np.isnan(tr_loss):
                        logging.warning('Epoch {0} -- step {1} ---- Loss went to np.nan.'
                                        .format(epoch, step_idx))
                        raise ValueError('np.nan error -- step {0}'.format(epoch))

                    tr_losses.append(tr_loss)
                    tr_metrix.append(tr_metric)
                    batch_ctr += 1

                # checkpoint model.
                if epoch in model_checkpoints:
                    logging.info('Checkpointing model.')
                    m.save(model_save_file.format(epoch=epoch, loss=tr_loss))

                # run validation set evaluation.
                if epoch in val_checkpoints:
                    if params['model_params']['validation_steps']:
                        val_metrix = evaluate_gen(val_gen
                                                  , model=m
                                                  , n_batches=params['model_params']['validation_steps'])
                        val_loss.append(val_metrix['loss'])
                        val_xtra_metric.append(val_metrix[params['model_params']['xtra_metric']])

                        logging.info('Epoch {0} ---- VALIDATION LOSS ---- {1}'
                                     .format(epoch, val_loss[-1]))

                        # evaluate early stopping criteria if a patience duration is specified.
                        if es:
                            if val_loss[-1] <= best_val_loss:
                                logging.info('New best validation set loss found!')
                                best_val_loss = val_loss[-1]
                                best_val_loss_epoch = epoch

                                # update best model.
                                m_best = m

                            else:
                                # determine how many validation checkpoints it's been since
                                # last improvement in validation set loss. Only do this
                                # if patience has expired.
                                val_idx = np.where(val_checkpoints == epoch)[0][0]
                                best_val_idx = np.where(val_checkpoints == best_val_loss_epoch)[0][0]
                                if val_idx - best_val_idx >= es:
                                    logging.info('Early stopping patience has expired! End model training.'
                                                 .format(epoch))
                                    break

                        # if not stopping early, just make the best model the current model.
                        else:
                            m_best = m

                # reset rnn state after validation set evaluation.
                m.reset_states()

                # get epoch-level mean loss and accuracy.
                mean_tr_loss = np.mean(tr_losses)
                mean_tr_metric = np.mean(tr_metrix)
                loss_history.append(mean_tr_loss)
                metrix_history.append(mean_tr_metric)
                logging.info('Epoch {0} ---- Training loss = {1}    {2} = {3}'
                             .format(epoch, mean_tr_loss, params['model_params']['xtra_metric'], mean_tr_metric))

            history = {'loss': loss_history
                       , params['model_params']['xtra_metric']: metrix_history}

            # consolidate validation set performance metrics into history dict.
            history['val_epochs'] = val_checkpoints[:len(val_loss)]
            history['val_loss'] = val_loss
            history['val_' + params['model_params']['xtra_metric']] = val_xtra_metric

        else:
            logging.info('Begin {0} model training...'.format(MODEL_TYPE))

            # set up keras model training progress logger callback.
            callbacks = [CSVLogger(os.path.join(output_dir, 'model_log.csv')
                                   , append=False
                                   , separator=';')
                         , TerminateOnNaN()
                         , ModelCheckpoint(model_save_file
                                           , period=params['model_params']['checkpoint_every'])]

            # if using early stopping, append EarlyStopping callback.
            if es:
                callbacks.append(EarlyStopping(monitor='loss'
                                               , patience=es))

            history = m.fit_generator(train_gen
                                      , steps_per_epoch=params['model_params']['steps_per_epoch']
                                      , epochs=params['model_params']['epochs']
                                      , validation_data=val_gen
                                      , validation_steps=params['model_params']['validation_steps']
                                      , shuffle=params['data_params']['shuffle']
                                      , class_weight=params['model_params']['class_weight']
                                      , verbose=2
                                      , callbacks=callbacks)

            history = history.history
            history['val_epochs'] = np.arange(1, params['model_params']['epochs'] + 1)

            # TODO: load up best model according to validation loss.
            m_best = m

        # plot model training/validation diagnostics and save that data.
        logging.info('Plotting training, validation history.')
        plot_training_history(history
                              , fname=os.path.join(output_dir, 'training_progress.png')
                              , metric_name=params['model_params']['xtra_metric']
                              , figsize=(8, 6))

        logging.info('Saving training, validation history.')
        with open(os.path.join(output_dir, 'model_history.pkl'), 'wb') as out_file:
            pkl.dump(history, out_file)

    else:
        logging.info('Loading pre-trained model from {0}'.format(args.pretrained_model))
        m_best = load_model(args.pretrained_model)

    # ---------------------------------------------------------------------- #
    # 6. Evaluate model on test chromosomes.
    # ---------------------------------------------------------------------- #

    # make model the best model according to validation set loss minimization.
    m = m_best
    m.reset_states()

    logging.info('BEGIN MODEL EVALUATION...')
    n_test_chroms = len(args.test_chromosomes)

    # subset all nucleosome data to test chromosomes.
    test_df = loader.subset_to_chromosomes(nuc_df
                                           , chromosomes=args.test_chromosomes).reset_index(drop=True)
    test_gen = gen_seq_scans(test_df
                             , lookback=params['data_params']['lookback']
                             , batch_size=params['data_params']['batch_size']
                             , target=TARGET
                             , target_position=params['data_params']['target_position']
                             , step_size=params['data_params']['output_len']
                             , flatten_final_target_dim=flatten
                             , consecutive_batches=True
                             , return_idx=True)

    evaluator = evaluate_binary_output_model
    if not BINARY:
        threshold = 0.05
        evaluator = partial(evaluate_cts_output_model
                            , threshold=threshold)

    # extract predictions from test chromosomes.
    logging.info('Extracting predictions for {0} test chromosomes...'.format(n_test_chroms))
    preds_df = extract_preds_from_test_set(test_gen
                                           , model=m
                                           , reset_data=test_df if stateful else None)

    logging.info('Extracted prediction data shape: {0}'.format(preds_df.shape))

    # Describe the distribution of predicted values.
    logging.info('(min, max) of predicted probabilities: ({:6.4f}, {:6.4f})'
                 .format(preds_df.preds.min(), preds_df.preds.max()))
    logging.info('mean of predicted probabilities = {:6.4f}'
                 .format(preds_df.preds.mean()))
    logging.info('standard deviation of predicted probabilities = {:6.4f}'
                 .format(preds_df.preds.std()))

    rand_idx = [int(i) for i in np.random.choice(range(preds_df.shape[0])
                                                 , size=3
                                                 , replace=False)]
    for i in range(len(rand_idx)):
        logging.info('Randomly sampled prediction {0} for inspection -- {1}'
                     .format(i + 1, preds_df.preds.iloc[rand_idx[i]]))

    # join test set data with predictions,
    # as we will need Chromosome, NCP, other data from test set during evaluation.
    test_df['idx'] = test_df.index.values
    test_df = test_df.merge(preds_df
                            , how='inner'
                            , on='idx')
    test_df.drop(['idx']
                 , axis=1
                 , inplace=True)

    logging.info('test_df.head():')
    logging.info(test_df.head())

    # Evaluate AUC/ROC performance for each chromosome.
    for chrom in args.test_chromosomes:
        chrom = 'chr' + str(chrom)
        out = evaluator(test_df[test_df.Chr == chrom]
                        , save_dir=output_dir
                        , chrom_name=chrom)

    # rename "preds" to nuclstm_preds, as we may be attaching NuPop and other types of predictions.
    test_df.rename(columns={'preds': 'nuclstm_preds'}
                   , inplace=True)

    # smooth out predictions.
    test_df = preprocess_by_chromosome(test_df
                                       , target='NCP/noise'
                                       , method='smooth'
                                       , smooth_window_len=147
                                       , smooth_window='hanning')

    # if binary model, locate the predicted nucleosome centers.
    if BINARY:
        logging.info('Finding predicted nucleosome centers...')
        test_df = locate_nucleosome_centers(test_df
                                            , pred_col='nuclstm_preds'
                                            , min_sep=params['postprocessor_params']['ncp_location']['min_nuc_sep']
                                            , min_max=params['postprocessor_params']['ncp_location']['min_max'])

        if 'nuclstm_preds_smoothed' not in test_corr_cols:
            test_corr_cols.append('nuclstm_preds_smoothed')

    # if NuPoP directory specified, join corresponding NuPoP predictions.
    if args.nupop_dir:
        logging.info('Joining nuclstm data with NuPoP output.')
        test_df = test_df.merge(nupop_df
                                , on=['Chr', 'pos'])
        check = check_dataframe_validity(test_df)

    # For each test chromosome:
    # (1) make sensitivity as a function of padding size plot
    # (2) render cross-correlation plot
    logging.info('Creating TPR / NCP distance and cross-correlation plots for test chromosomes.')
    for chrom in args.test_chromosomes:
        chrom = 'chr' + str(chrom)
        test_chrom_df = test_df[test_df.Chr == chrom]

        sensitivity_kdist_plot(test_chrom_df
                               , fname=os.path.join(output_dir, 'sensitivity_' + chrom + '.png')
                               , preds1='nupop_ncp'
                               , preds2='nuclstm_ncp')

        plot_cross_correlation(test_chrom_df.nucleosome.values
                               , preds=test_chrom_df.nuclstm_ncp.values
                               , fname=os.path.join(output_dir
                                                    , 'cross_correlation_' + str(chrom) + '.png')
                               , lag_width=80)

    if 'Occup' not in test_corr_cols:
        test_corr_cols.append('Occup')

    # create and save correlation heatmaps from desired fields for.
    logging.info('Saving correlation heatmaps.')
    plot_correlation_heatmap(test_df[test_corr_cols]
                             , fname=os.path.join(output_dir
                                                  , 'corr_heatmap.png')
                             , figsize=(10, 10))

    # save test set and preds.
    logging.info('Saving test set + predictions data ---- shape = {0}'
                 .format(test_df.shape))
    test_df.to_csv(os.path.join(output_dir, 'test_preds.csv')
                   , index=False)

    # exit.
    logging.info('{0} MODEL TRAINING AND EVALUATION COMPLETE. EXITING...'.format(MODEL_NAME))
    sys.exit(0)