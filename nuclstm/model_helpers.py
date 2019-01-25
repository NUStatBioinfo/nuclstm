from keras.models import load_model
from sklearn.utils import class_weight
import keras.backend as K
import numpy as np
import re
import os


def get_background_entropy(x):
    p = np.mean(x)
    return -1 * (p * np.log(p) + (1 - p) * np.log(1 - p))


def normalized_binary_ce(y_true, y_pred, background):
    """
    Normalized binary crossentropy loss function. Make crossentropy
    impervious to highly imbalanced binary target vectors by dividing loss
    by a background entropy.  See https://bit.ly/2HPbP7C for details.
    """
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) / background


def load_keras_model(model_dir, model_file_pattern='^model.*.h5$'):
    """
    Load the first file in model_dir that matches the pattern model_file_pattern.
    """
    model_files = [f for f in os.listdir(model_dir) if re.match(model_file_pattern, f)]
    m = load_model(model_files[0])

    return m


def determine_xtra_metric(loss):
    """
    Given the name of a loss function, determine likely useful additional metric
    to track during training.

    :param loss: str name of loss function being optimized
    :return: str name of additional useful metric to track
    """
    if loss == 'mae':
        xtra_metric = 'mse'
    elif loss == 'mse':
        xtra_metric = 'mae'
    elif loss in ['binary_crossentropy', 'normalized_binary_cross', 'softmax']:
        xtra_metric = 'acc'
    else:
        raise ValueError('loss function `{0}` not paired with a supplemental metric.'.format(loss))

    return xtra_metric


def validate_model_type(model_type):
    """
    Check that a model type is valid.

    :param model_type: str name of a type of model, e.g. "lstm"
    :return: Boolean True
    :raises: NotImplementedError if type of model not yet implemented
    """
    model_type = model_type.lower()
    valid_types = ['dense', 'rnn', 'covrnn']

    if model_type not in valid_types:
        raise NotImplementedError('model type {0} is not yet a valid model type.'.format(model_type))

    return model_type


def get_binary_class_weights(x):
    """
    Compute balanced class weights for a binary vector.

    :param x: np.array or list of binary values
    :return: dictionary of class weightings
    """
    if len(np.unique(x)) != 2:
        raise ValueError('x must be a vector of binary class indicators..')

    cw = class_weight.compute_class_weight('balanced'
                                           , [0, 1]
                                           , x)
    cw = {0: cw[0], 1: cw[1]}

    return cw


def evaluate_gen(gen, model, n_batches=5):
    """
    Evaluate a keras model with a data generator, but over a set of batches.

    :param gen: data generator that serves batches of x and y data
    :param model: a compiled Keras model
    :param n_batches: number of batches to evaluate. Take average metric performance over batches.
    :return: dictionary of form {`metric name`: `avg metric value`}
    """
    n_metrics = len(model.metrics_names)
    metrics_arr = np.zeros([n_batches, n_metrics])

    for i in range(n_batches):

        x_tmp, y_tmp = next(gen)
        metrics_arr[i, :] = model.evaluate(x_tmp
                                           , y=y_tmp
                                           , batch_size=x_tmp.shape[0]
                                           , verbose=0)

    performance = dict(zip(model.metrics_names, np.mean(metrics_arr
                                                        , axis=0)))

    return performance
