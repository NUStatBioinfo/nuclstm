from keras.models import Sequential
from keras.layers import *
from keras.regularizers import l1_l2
from loaders import *
from model_helpers import *
from functools import partial
import os


class ModelBuilder():

    def __init__(self, config_file=None, config=None):

        if isinstance(config, dict):
            self.config = config
        elif config_file:
            self.config = ConfigLoader(config_file).load()
        else:
            raise ValueError('Must specify either config_file or config (a dictionary).')

        self.model_type = validate_model_type(self.config['run_params']['model_type'])
        self.lookback = self.config['data_params']['lookback']
        self.batch_size = self.config['data_params']['batch_size']
        self.output_len = self.config['data_params']['output_len']

        self.xtra_metric = self.config['model_params'].get('xtra_metric'
                                                           , determine_xtra_metric(self.config['optimizer_params']['loss']))


    @staticmethod
    def int_2_list(x):
        if isinstance(x, list):
            return x
        else:
            return [x]


    def build_rnn(self, covnet=None):

        rnn_params = self.config['model_params']['rnn_params']

        hidden_units = self.int_2_list(rnn_params['hidden_units'])
        n_hidden_layers = len(hidden_units)

        # if not timedistributed and we're on the last hidden layer in the network: return_sequences = False
        timedistributed = self.config['model_params']['dense_params']['timedistributed']
        stateful = rnn_params['stateful']
        bidirectional = rnn_params['bidirectional']

        # define main hidden layer
        intermediate_layer = partial(LSTM if rnn_params['flavor'] == 'lstm' else GRU
                                     , activation=rnn_params['activation']
                                     , use_bias=True
                                     , return_sequences=True
                                     , stateful=rnn_params['stateful'])

        if not timedistributed:
            final_hidden_layer = partial(LSTM if rnn_params['flavor'] == 'lstm' else GRU
                                         , activation=rnn_params['activation']
                                         , use_bias=True
                                         , stateful=rnn_params['stateful'])
        else:
            final_hidden_layer = intermediate_layer

        if rnn_params.get('initializer', None):
            intermediate_layer = partial(intermediate_layer, kernel_initializer=rnn_params.get('initializer'))
            final_hidden_layer = partial(final_hidden_layer, kernel_initializer=rnn_params.get('initializer'))

        # initialize either an RNN or a covnet-RNN hybrid.
        m = covnet if covnet else Sequential()

        for i in range(n_hidden_layers):
            neurons = hidden_units[i]
            dropout = rnn_params['dropout'][i]
            recurrent_dropout = rnn_params['recurrent_dropout'][i]
            reg_penalty = rnn_params['regularization'][i]

            if i == 0:
                l = intermediate_layer if n_hidden_layers > 1 else final_hidden_layer

                if stateful:
                    if not covnet:
                        if bidirectional:
                            m.add(Bidirectional(l(units=neurons
                                                  , dropout=dropout
                                                  , recurrent_dropout=recurrent_dropout
                                                  , kernel_regularizer=l1_l2(reg_penalty, reg_penalty))
                                                , batch_input_shape=(self.batch_size, self.lookback, 4)))
                        else:
                            m.add(l(units=neurons
                                    , dropout=dropout
                                    , recurrent_dropout=recurrent_dropout
                                    , kernel_regularizer=l1_l2(reg_penalty, reg_penalty)
                                    , batch_input_shape=(self.batch_size, self.lookback, 4)))
                    else:
                        if bidirectional:
                            m.add(Bidirectional(l(units=neurons
                                                  , dropout=dropout
                                                  , recurrent_dropout=recurrent_dropout
                                                  , kernel_regularizer=l1_l2(reg_penalty, reg_penalty))))
                        else:
                            m.add(l(units=neurons
                                    , dropout=dropout
                                    , recurrent_dropout=recurrent_dropout))
                else:
                    if not covnet:
                        if bidirectional:
                            m.add(Bidirectional(l(units=neurons
                                                  , dropout=dropout
                                                  , recurrent_dropout=recurrent_dropout
                                                  , kernel_regularizer=l1_l2(reg_penalty, reg_penalty))
                                                , input_shape=(self.lookback, 4)))
                        else:
                            m.add(l(units=neurons
                                    , dropout=dropout
                                    , recurrent_dropout=recurrent_dropout
                                    , kernel_regularizer=l1_l2(reg_penalty, reg_penalty)
                                    , input_shape=(self.lookback, 4)))
                    else:
                        if bidirectional:
                            m.add(Bidirectional(l(units=neurons
                                                  , dropout=dropout
                                                  , recurrent_dropout=recurrent_dropout
                                                  , kernel_regularizer=l1_l2(reg_penalty, reg_penalty))))
                        else:
                            m.add(l(units=neurons
                                    , dropout=dropout
                                    , recurrent_dropout=recurrent_dropout
                                    , kernel_regularizer=l1_l2(reg_penalty, reg_penalty)))

            else:
                l = intermediate_layer if i < (n_hidden_layers - 1) else final_hidden_layer

                if bidirectional:
                    m.add(Bidirectional(l(units=neurons
                                          , dropout=dropout
                                          , recurrent_dropout=recurrent_dropout
                                          , kernel_regularizer=l1_l2(reg_penalty, reg_penalty))))
                else:
                    m.add(l(units=neurons
                            , dropout=dropout
                            , recurrent_dropout=recurrent_dropout
                            , kernel_regularizer=l1_l2(reg_penalty, reg_penalty)))

        return m


    def build_dense(self):

        dense_params = self.config['model_params']['dense_params']

        hidden_units = self.int_2_list(dense_params['hidden_units'])

        m = Sequential()
        m.add(Flatten(input_shape=(self.lookback, 4)))

        for i in len(hidden_units):
            reg_penalty = dense_params['regularization'][i]
            m.add(Dense(hidden_units[i]
                        , activation=dense_params['activation']
                        , kernel_regularizer=l1_l2(reg_penalty, reg_penalty)
                        , use_bias=True))

        return m


    def build_covrnn(self):

        covnet_params = self.config['model_params']['covnet_params']

        filters = self.int_2_list(covnet_params['filters'])
        kernel_sizes = self.int_2_list(covnet_params['kernel_sizes'])
        pool_sizes = self.int_2_list(covnet_params['pool_sizes'])
        strides = self.int_2_list(covnet_params['strides'])

        m = Sequential()

        for i in range(len(filters)):
            cov_layer = partial(Conv1D
                                , filters=filters[i]
                                , kernel_size=kernel_sizes[i]
                                , strides=strides[i]
                                , padding='valid'
                                , activation=covnet_params['activation'])
            if i == 0:
                if self.config['model_params']['rnn_params']['stateful']:
                    cov_layer = partial(cov_layer
                                        , batch_input_shape=(self.batch_size, self.lookback, 4))
                else:
                    cov_layer = partial(cov_layer
                                        , input_shape=(self.lookback, 4))

            if covnet_params.get('initializer', None):
                cov_layer = partial(cov_layer, kernel_initializer=covnet_params.get('initializer'))

            m.add(cov_layer())
            m.add(MaxPooling1D(pool_sizes[i]))

        m = self.build_rnn(covnet=m)

        return m


    def build(self):

        if self.model_type == 'rnn':
            m = self.build_rnn()
        elif self.model_type == 'covrnn':
            m = self.build_covrnn()
        elif self.model_type == 'dense':
            m = self.build_dense()

        # figure out how many neurons comprise the final dense layer of the model.
        timedistributed = self.config['model_params']['dense_params']['timedistributed']
        final_neurons = self.output_len

        # construct the model's output layer.
        if self.config['data_params']['target_position'] == 'all' and self.model_type == 'rnn':
            final_neurons = 1

        reg_dict = self.config['model_params']['dense_params']['regularization']
        reg_penalty = reg_dict[max(reg_dict.keys())]
        if self.config['model_params']['dense_params'].get('initializer', None):
            dense_layer = Dense(final_neurons
                                , activation=self.config['model_params']['final_layer_activation']
                                , kernel_initializer=self.config['model_params']['dense_params']['initializer']
                                , kernel_regularizer=l1_l2(reg_penalty, reg_penalty)
                                , use_bias=True)
        else:
            dense_layer = Dense(final_neurons
                                , activation=self.config['model_params']['final_layer_activation']
                                , kernel_regularizer=l1_l2(reg_penalty, reg_penalty)
                                , use_bias=True)

        if timedistributed:
            m.add(TimeDistributed(dense_layer))
        else:
            m.add(dense_layer)

        m.compile(loss=self.config['optimizer_params']['loss']
                  , optimizer=self.config['optimizer_params']['optimizer']
                  , metrics=[self.xtra_metric])

        return m


if __name__ == '__main__':

    config_file = os.path.join(os.getenv('HOME')
                               , 'jiping_research'
                               , 'nuclstm'
                               , 'model_configs'
                               # , 'lstm_binary_sequence_config.json')
                                , 'gru_binary_sequence_config.json')
                               # , 'dense_baseline_config.json')
                               #, 'covlstm_stateless_binary_scalar_config.json')
    #from keras import backend as K
    #K.set_floatx('float32')

    model = ModelBuilder(config_file).build()

    print(model.summary())
