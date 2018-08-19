import tensorflow as tf


def delta_weights(v0, h0, vn, hn, log_sigmas=None):
    """
    calculate update for weights
    :param v0: initial values of visible units, shape: [batch_size, num_vunits]
    :param h0: values of hidden units after one gibbs step, shape: [batch_size, num_hunits]
    :param vn: values of visible units after running markov chain, shape: [batch_size, num_vunits]
    :param hn: values of hidden units after running markov chain, shape: [batch_size, num_hunits]
    :param log_sigmas: logarithmic squared standard deviation (only when using Gaussian units), shape: [1, num_vunits]
    :return: weight update, shape: [num_vunits, num_hunits]
    """
    delta_w = (tf.matmul(v0, h0, transpose_a=True) -
               tf.matmul(vn, hn, transpose_a=True)) / tf.cast(tf.shape(v0)[0], tf.float32)
    if log_sigmas is not None:
        delta_w = tf.divide(delta_weights, tf.transpose(tf.exp(log_sigmas)))
    return delta_w


def delta_vbiases(v0, vn, log_sigmas=None):
    """
    calculate update for visual biases
    :param v0: initial values of visible units, shape: [batch_size, num_vunits]
    :param vn: values of visible units after running markov chain, shape: [batch_size, num_vunits]
    :param log_sigmas: logarithmic squared standard deviation (only when using Gaussian units), shape: [1, num_vunits]
    :return: visual bias update, shape: [1, num_vunits]
    """
    delta_vb = tf.reduce_mean(v0 - vn, axis=0, keepdims=True)
    if log_sigmas is not None:
        delta_vb = tf.divide(delta_vbiases, tf.exp(log_sigmas))
    return delta_vb


def delta_hbiases(h0, hn):
    """
    calculate update for hidden biases
    :param h0: values of hidden units after one gibbs step, shape: [batch_size, num_hunits]
    :param hn: values of hidden units after running markov chain, shape: [batch_size, num_hunits]
    :return: hidden bias update, shape: [1, num_hunits]
    """
    return tf.reduce_mean(h0 - hn, axis=0, keepdims=True)


def delta_log_sigmas(v0, h0, vn, hn, vbiases, weights, log_sigmas):
    """
    calculate update for logarithmic squared standard deviation (only when using Gaussian units)
    :param v0: initial values of visible units, shape: [batch_size, num_vunits]
    :param h0: values of hidden units after one gibbs step, shape: [batch_size, num_hunits]
    :param vn: values of visible units after running markov chain, shape: [batch_size, num_vunits]
    :param hn: values of hidden units after running markov chain, shape: [batch_size, num_hunits]
    :param vbiases: visual biases, shape: [1, num_vunits]
    :param weights: hidden biases, shape: [1, num_hunits]
    :param log_sigmas: logarithmic squared standard deviation, shape: [1, num_vunits]
    :return: log sigmas update, shape: [1, num_vunits]
    """
    data_term = tf.reduce_mean(0.5*tf.square(v0 - vbiases)
                               - tf.multiply(v0, tf.transpose(tf.matmul(weights, h0, transpose_b=True))),
                               axis=0, keepdims=True)

    model_term = tf.reduce_mean(0.5*tf.square(vn - vbiases)
                                - tf.multiply(vn, tf.transpose(tf.matmul(weights, hn, transpose_b=True))),
                                axis=0, keepdims=True)

    delta_ls = tf.multiply(tf.exp(-log_sigmas), (data_term - model_term))
    return delta_ls


def error(v0, vn):
    """
    calculate reconstruction error
    :param v0: initial values of visible units, shape: [batch_size, num_vunits]
    :param vn: values of visible units after running markov chain, shape: [batch_size, num_vunits]
    :return: reconstruction error (scalar)
    """
    return tf.reduce_mean(tf.square(v0 - vn))


def sparse_terms(p, q, h_, v0, log_sigmas=None):
    """
    calculate sparsity terms for the parameter updates (from cross-entropy between
                                                        goal activation and actual activation)
    :param p: goal activation(-probability) of the hidden units (scalar)
    :param q: mean activation(-probabilities) over one batch of hidden units, shape:[1, num_hunits]
    :param h_: activation(-probabilities) of hidden units, shape:[batch_size, num_hunits]
    :param v0: initial values of the visible units, shape:[batch_size, num_vunits]
    :param log_sigmas:  logarithmic squared standard deviation (only when using Gaussian units),
     shape: [1, num_vunits]
    :return: sparsity term for hidden biases update and for weights update
    """
    sparse_term_hbias = q - p
    if log_sigmas is not None:
        v0 = tf.divide(v0, tf.exp(log_sigmas))
    sparse_term_w = tf.matmul(v0, (h_ - p), transpose_a=True) / tf.cast(tf.shape(v0)[0], tf.float32)
    return sparse_term_hbias, sparse_term_w
