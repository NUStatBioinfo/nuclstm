import os
import pandas as pd
import feather
import json
from keras import optimizers
import pprint
from utils import *


class NcpLoader():
    def __init__(self, data_dir):
        """
        Nucleosome center positioning (NCP) data loader. See scripts/nuclstm_data_generation.py
        for script that generates the full genome redundant or unique nucleosome map datasets.

        :param data_dir: str path (real or relative) to directory where
        a .feather file containing full genome redundant or unique nucleosome map data lives.
        """
        if not os.path.isdir(data_dir):
            raise ValueError('{0} is not a valid directory'.format(data_dir))

        self.data_dir = data_dir

    def load(self, feather_file, chromosomes=None):
        """
        Load a dataset of genome NCP + sequence + binary nucleosome occupancy data stored in a feather file,
        and subset to a specific set of chromosomes if desired.

        Contents of .feather file should resemble the following:


        See scripts/nuclstm_data_generation.py for the origin of such a dataset.

        :param feather_file: str filename of .feather file containing genome sequence,
        NCP score, and nucleosome indicator data. See self.create_ncp_dataset.
        :param chromosomes: list of int, chromosome numbers to load in
        :return: pandas.DataFrame with at least ['Chr', 'pos', 'NCP', 'NCP/noise', 'nucleosome'] columns.
        """

        # check ncp_feather_file path validity.
        if not os.path.isfile(os.path.join(self.data_dir, feather_file)):
            raise IOError('{0} file was not found within {1}'.format(feather_file, self.data_dir))

        # parse and order chromosomes, if supplied. Else assume all 16.
        if chromosomes:
            if not isinstance(chromosomes, list):
                chromosomes = [chromosomes]

            chromosomes = list(set(chromosomes))
            chromosomes.sort()

        # read in data created from self.create_ncp_dataset, subset to desired chromosomes.
        nuc_df = feather.read_dataframe(os.path.join(self.data_dir, feather_file))

        if chromosomes:
            nuc_df = self.subset_to_chromosomes(nuc_df
                                                , chromosomes=chromosomes)

        # reset index after subset.
        nuc_df.reset_index(drop=True
                           , inplace=True)

        if 'index' in nuc_df.columns:
            nuc_df.drop('index'
                        , axis=1
                        , inplace=True)

        # check that we actually have data.
        check = check_dataframe_validity(nuc_df)

        return nuc_df

    def subset_to_chromosomes(self, df, chromosomes):
        """
        Subset a pandas.DataFrame with chromosomes listed in roman numerals
        by a set of integer-numbered chromosomes.

        :param df: pandas.DataFrame
        :param chromosomes: int or list of int specifying desired chromosome subset
        :return: pandas.DataFrame subset of original df
        """
        check = check_dataframe_validity(df
                                         , reqd_cols=['Chr'])

        if isinstance(chromosomes, int):
            chromosomes = [chromosomes]
        sub_df = df[df['Chr'].isin(['chr' + str(x) for x in chromosomes])]

        check = check_dataframe_validity(sub_df)

        return sub_df.reset_index(drop=True)

    @staticmethod
    def load_nupop_data(nupop_file):
        """
        Load NuPoP results from a .feather file for the S. cerevisiae genome

        :param nupop_file: str .feather file name
        :return: pd.DataFrame
        """
        if not os.path.isfile(os.path.join(nupop_file)):
            raise IOError('{0} file not found'.format(nupop_file))
        if not nupop_file.endswith('.feather'):
            raise ValueError('{0} should be a .feather file'.format(nupop_file))

        nupop_df = feather.read_dataframe(nupop_file)
        int_cols = ['pos', 'N/L', 'nupop_ncp']
        nupop_df[int_cols] = nupop_df[int_cols].astype(int)

        float_cols = list(set(nupop_df.columns.tolist()) - set(int_cols + ['Chr']))
        for col in float_cols:
            nupop_df.loc[nupop_df[col] == 'NA', col] = np.nan
            nupop_df[col] = nupop_df[col].astype(float)

        check = check_dataframe_validity(nupop_df)

        return nupop_df


class ConfigLoader():

    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise IOError('{0} config file not found!'.format(config_file))

        self.config_file = config_file

    def parse_rnn_layers(self, d):
        settings = dict()
        if not d['flavor'].lower() in ['lstm', 'gru']:
            raise ValueError('flavor must be either `lstm` or `gru`')
        else:
            settings['flavor'] = d['flavor'].lower()

        settings['hidden_units'] = d['hidden_units']
        n_hidden_layers = len(settings['hidden_units'])

        # handle dropout, regularization settings with a default of 0% dropout, 0 weight penalty.
        default_dict = dict(zip(list(range(n_hidden_layers)), [0.0]*n_hidden_layers))
        for drp in ['dropout', 'recurrent_dropout', 'regularization']:
            drp_reg_dict = d.get(drp, default_dict)
            drp_reg_dict = {int(k): float(v) for k, v in drp_reg_dict.items()}

            for i in range(n_hidden_layers):
                if i not in drp_reg_dict:
                    drp_reg_dict[i] = 0.0

            settings[drp] = drp_reg_dict

        settings['activation'] = d.get('activation', 'tanh')
        settings['stateful'] = d.get('stateful', False)
        settings['bidirectional'] = d.get('bidirectional', False)
        if settings.get('initializer', None):
            settings['initializer'] = eval(settings['initializer'])

        return settings

    def parse_covnet_layers(self, d):
        settings = dict()
        settings['filters'] = d['filters']
        settings['kernel_sizes'] = d['kernel_sizes']
        settings['pool_sizes'] = d['pool_sizes']

        if not len(settings['filters']) == len(settings['kernel_sizes']):
            raise ValueError('Every covnet layer must have a number of filters and kernel_size associated with it.')

        if not len(settings['filters']) <= len(settings['pool_sizes']):
            raise ValueError('Every covnet layer must have a specified pooling size for 1-d max pooling.')

        settings['activation'] = d.get('activation', 'relu')
        settings['strides'] = d.get('strides', [1]*len(settings['kernel_sizes']))
        if settings.get('initializer', None):
            settings['initializer'] = eval(settings['initializer'])

        return settings

    def parse_dense_layers(self, d):
        settings = dict()
        settings['hidden_units'] = d.get('hidden_units', 1)
        settings['timedistributed'] = d.get('timedistributed', False)
        settings['activation'] = d['activation']
        if settings.get('initializer', None):
            settings['initializer'] = eval(settings['initializer'])

        # handle regularization settings with a default of 0 weight penalty.
        default_dict = dict(zip(list(range(settings['hidden_units'])), [0.0] * settings['hidden_units']))
        reg_dict = d.get('regularization', default_dict)
        reg_dict = {int(k): float(v) for k, v in reg_dict.items()}
        settings['regularization'] = reg_dict

        return settings

    def load(self):
        settings = json.load(open(self.config_file))
        default_model_name = os.path.split(self.config_file)[1].replace('_config.json', '')

        # parse general model run configuration details.
        run_params = {'model_name': settings['run_params'].get('model_name', default_model_name)
                      , 'model_type': settings['run_params'].get('model_type', None)
                      , 'binary': settings['run_params'].get('binary', True)
                      , 'floatx': settings['run_params'].get('floatx', None)}

        # parse parameters related to optimizer, objective.
        model_params = {
            'epochs': settings['model_params'].get('epochs', 100)
            , 'steps_per_epoch': settings['model_params'].get('steps_per_epoch', 500)
            , 'validation_steps': settings['model_params'].get('validation_steps', 10)
            , 'xtra_metric': settings['model_params'].get('xtra_metric', None)
            , 'target': settings['model_params'].get('target', 'nucleosome')
            , 'early_stopping': settings['model_params'].get('early_stopping', None)
            , 'checkpoint_every': int(settings['model_params'].get('checkpoint_every', 20))
            , 'validate_every': int(settings['model_params'].get('validate_every', 20))}

        # parse model architecture depending on layers in config.
        if settings['model_params'].get('covnet_params', None):
            model_params['covnet_params'] = self.parse_covnet_layers(settings['model_params']['covnet_params'])

        if settings['model_params'].get('rnn_params', None):
            model_params['rnn_params'] = self.parse_rnn_layers(settings['model_params']['rnn_params'])

        if settings['model_params'].get('dense_params', None):
            model_params['dense_params'] = self.parse_dense_layers(settings['model_params']['dense_params'])

        # determine activation of the final layer.
        default_final_activation = 'sigmoid'
        if model_params.get('dense_params', None):
            default_final_activation = model_params.get('dense_params').get('activation', default_final_activation)
        model_params['final_layer_activation'] = settings['model_params'].get('final_layer_activation'
                                                                              , default_final_activation)

        # parse optimizer settings: evaluate `call`, which should create a keras.optimizers object.
        optimizer_params = {'optimizer': eval(settings['optimizer_params']['call'])
                            , 'loss': settings['optimizer_params'].get('loss', 'binary_crossentropy')}

        # parse data settings: to be used for configuring data generators.
        data_params = {'lookback': settings['data_params'].get('lookback', 250)
            , 'shuffle': settings['data_params'].get('shuffle', False)
            , 'batch_size': settings['data_params'].get('batch_size', 128)
            , 'no_one_class_samples': settings['data_params'].get('no_one_class_samples', False)}

        # default step between samples in a batch is the size of the sample.
        data_params['step_size'] = settings['data_params'].get('step_size', data_params['lookback'])

        # parse preprocessing steps.
        preprocessor_params = dict()
        if settings.get('preprocessor_params', None):
            if settings['preprocessor_params'].get('target', None):
                preprocessor_params['target'] = {
                    'method': settings['preprocessor_params']['target']['method']
                    , 'smooth_window': settings['preprocessor_params']['target'].get('smooth_window', None)
                    , 'smooth_window_len': settings['preprocessor_params']['target'].get('smooth_window_len', None)
                    , 'pad_len': settings['preprocessor_params']['target'].get('pad_len', None)}

        # figure out if we're doing class weighting in the loss function.
        class_weight = None
        if 'class_weight' in settings['model_params']:
            class_weight = settings['model_params']['class_weight']

            if isinstance(class_weight, dict):
                for c in class_weight:
                    class_weight[int(c)] = class_weight.pop(c)

        model_params['class_weight'] = class_weight

        # determine the length of the output target vector
        if 'output_len' in settings['data_params']:
            try:
                output_len = split_comma_ints(settings['data_params']['output_len'])
                if len(output_len) == 2:
                    output_len = list(range(output_len[0], output_len[1]))
                if len(output_len) == 1:
                    output_len = output_len[0]
            except:
                output_len = 1
        else:
            output_len = 1

        if 'target_position' in settings['data_params']:
            try:
                target_position = split_comma_ints(settings['data_params']['target_position'])
                if len(target_position) == 2:
                    target_position = list(range(target_position[0], target_position[1]))
                if len(target_position) == 1:
                    target_position = target_position[0]

            except ValueError:
                target_position = settings['data_params']['target_position']
        else:
            target_position = 'median'
            output_len = 1

        if target_position == 'all':
            output_len = data_params['lookback']

        if not isinstance(target_position, str) and len(target_position) > 1:
            output_len = len(target_position)

        data_params['target_position'] = target_position
        data_params['output_len'] = output_len

        # if predicting a scalar, timedistributed cannot be set to True.
        if output_len == 1:
            model_params['dense_params']['timedistributed'] = False

        # assemble postprocessor settings and defaults.
        postprocessor_params = {
            'ncp_location': {
                'min_max': 0.0
                , 'min_nuc_sep': 127
            }
        }
        if settings.get('postprocessor_params'):
            if settings['postprocessor_params'].get('ncp_location'):
                postprocessor_params['ncp_location'] = {
                    'min_max': settings['postprocessor_params']['ncp_location'].get('min_max'
                                                                                    , postprocessor_params['ncp_location']['min_max'])
                    , 'min_nuc_sep': settings['postprocessor_params']['ncp_location'].get('min_nuc_sep'
                                                                                          , postprocessor_params['ncp_location']['min_nuc_sep'])}

        # concatenate all the parameter dicts into one.
        all_params = dict()
        all_params['run_params'] = run_params
        all_params['model_params'] = model_params
        all_params['optimizer_params'] = optimizer_params
        all_params['data_params'] = data_params
        all_params['postprocessor_params'] = postprocessor_params

        if preprocessor_params:
            all_params['preprocessor_params'] = preprocessor_params

        return all_params


if __name__ == '__main__':
    # pd.set_option('display.max_rows', 500)
    #
    # loader = NcpLoader(os.path.join(os.getenv('HOME')
    #                                 , 'nu'
    #                                 , 'jiping_research'
    #                                 , 'data'))
    #
    # df = loader.load('unique_nucleosome_map.feather'
    #                  , chromosomes=list(range(1, 10)))
    #
    # print(df.head())
    # print('loaded nucleosome dataset shape: {0}'.format(df.shape))
    # print('loaded nucleosome dataset chromosomes: {0}'.format(len(df.Chr.unique())))
    # print('loaded nucleosome dataset nucleosomes: {0}'.format(df.nucleosome.sum()))

    config_file = os.path.join(os.getenv('HOME')
                               , 'nu'
                               , 'jiping_research'
                               , 'nuclstm'
                               , 'model_configs'
                               , 'gru_binary_sequence_config.json')

    config = ConfigLoader(config_file).load()

    pprint.pprint(config)