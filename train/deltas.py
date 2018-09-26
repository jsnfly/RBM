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
        delta_w = tf.divide(delta_w, tf.transpose(tf.exp(log_sigmas)))
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
        delta_vb = tf.divide(delta_vb, tf.exp(log_sigmas))
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
    data_term = tf.reduce_mean(0.5 * tf.square(v0 - vbiases)
                               - tf.multiply(v0, tf.transpose(tf.matmul(weights, h0, transpose_b=True))),
                               axis=0, keepdims=True)

    model_term = tf.reduce_mean(0.5 * tf.square(vn - vbiases)
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


def enhanced_deltas(v0, h0, vn, hn, batch_size):
    """
    calculate learning step for weights
    Args:
        v0: array of initial values of visible units, shape: [batch_size,num_vUnits]
        h0: array of values of hidden units after one gibbs step, shape: [batch_size,num_hUnits]
        vn: array of values of visible units after running markov chain, shape: [batch_size,num_vUnits]
        hn: array of values of hidden units after running markov chain, shape: [batch_size,num_hUnits]
    Returns:
        delta vbiases, shape: [1,num_vUnits]
        delta hbiases, shape: [1,num_hUnits]
        delta weights, shape: [num_vUnits,num_hUnits]
    """
    avg_vh_data = tf.matmul(v0, h0, transpose_a=True) / batch_size
    avg_vh_model = tf.matmul(vn, hn, transpose_a=True) / batch_size
    avg_v_data = tf.reduce_mean(v0, axis=0, keepdims=True)
    avg_v_model = tf.reduce_mean(vn, axis=0, keepdims=True)
    avg_h_data = tf.reduce_mean(h0, axis=0, keepdims=True)
    avg_h_model = tf.reduce_mean(hn, axis=0, keepdims=True)

    delta_w = avg_vh_data - avg_vh_model - tf.matmul(avg_v_data, avg_h_data, transpose_a=True) + tf.matmul(avg_v_model,
                                                                                                           avg_h_model,
                                                                                                           transpose_a=True)
    delta_vb = avg_v_data - avg_v_model - tf.matmul(0.5 * (avg_h_data + avg_h_model), delta_w, transpose_b=True)
    delta_hb = avg_h_data - avg_h_model - tf.matmul(0.5 * (avg_v_data + avg_v_model), delta_w)
    return delta_vb, delta_hb, delta_w


def enhanced_delta_log_sigmas(v0, h0, v1, h1, vbiases, weights, log_sigma):
    """
    calculate probabilities for hidden units to be on given gaussian visible units
    Args:
        v0: array of initial values of visible units, shape: [batch_size,num_vUnits]
        h0: array of values of hidden units after one gibbs step, shape: [batch_size,num_hUnits]
        v1: array of values of visible units after running markov chain, shape: [batch_size,num_vUnits]
        h1: array of values of hidden units after running markov chain, shape: [batch_size,num_hUnits]
        weights: weight matrix, shape: [num_vUnits,num_hUnits]
        vbiases: biases of visible units, shape: [1,num_vUnits]
        log_sigma: log of squared standard deviations of visible units, shape: [1,num_vUnits]
    Returns:
        delta for log sigmas
    """
    data_term = tf.reduce_mean(0.5 * tf.square(v0 - vbiases)
                               - tf.multiply(v0, tf.transpose(tf.matmul(weights, h0, transpose_b=True))),
                               axis=0, keepdims=True)
    model_term = tf.reduce_mean(0.5 * tf.square(v1 - vbiases)
                                - tf.multiply(v1, tf.transpose(tf.matmul(weights, h1, transpose_b=True))),
                                axis=0, keepdims=True)
    delta_ls = tf.multiply(tf.exp(-log_sigma), (data_term - model_term))
    return delta_ls
