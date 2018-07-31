import tensorflow as tf


def delta_weights(v0, h0, vn, hn, log_sigma=None):
    '''
    calculate learning step for weights
    Args:
        v0: array of initial values of visible units, shape: [batch_size,num_vUnits]
        h0: array of values of hidden units after one gibbs step, shape: [batch_size,num_hUnits]
        v1: array of values of visible units after running markov chain, shape: [batch_size,num_vUnits]
        h1: array of values of hidden units after running markov chain, shape: [batch_size,num_hUnits]
    Returns:
        delta weights, shape: [num_vUnits,num_hUnits]
    '''
    delta_weights = (tf.matmul(v0, h0, transpose_a=True) -
                     tf.matmul(vn, hn, transpose_a=True)) / tf.cast(tf.shape(v0)[0], tf.float32)
    if log_sigma != None:
        delta_weights = tf.divide(delta_weights, tf.transpose(tf.exp(log_sigma)))
    return delta_weights


def delta_vbiases(v0, v1, log_sigma=None):
    '''
    calculate learning step for visual biases
    Args:
        v0: array of initial values of visible units, shape: [batchsize,num_vUnits]
        v1: array of values of visible units after running markov chain, shape: [batchsize,num_vUnits]
    Returns:
        delta vbiases, shape: [1,num_vUnits]
    '''
    delta_vbiases = tf.reduce_mean(v0 - v1, axis=0, keepdims=True)
    if log_sigma != None:
        delta_vbiases = tf.divide(delta_vbiases, tf.exp(log_sigma))
    return delta_vbiases


def delta_hbiases(h0, h1):
    '''
    calculate learning step for hidden biases
    Args:
        h0: array of values of hidden units after one gibbs step, shape: [batchsize,num_hUnits]
        h1: array of values of hidden units after running markov chain, shape: [batchsize,num_hUnits]
    Returns:
        delta hbiases, shape: [1,num_hUnits]
    '''
    return tf.reduce_mean(h0 - h1, axis=0, keepdims=True)


def delta_log_sigmas(v0, h0, v1, h1, vbiases, weights, log_sigma):
    '''
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
    '''
    data_term = tf.reduce_mean(0.5 * tf.square(v0 - vbiases)
                               - tf.multiply(v0, tf.transpose(tf.matmul(weights, h0, transpose_b=True))),
                               axis=0, keepdims=True)
    model_term = tf.reduce_mean(0.5 * tf.square(v1 - vbiases)
                                - tf.multiply(v1, tf.transpose(tf.matmul(weights, h1, transpose_b=True))),
                                axis=0, keepdims=True)
    delta_log_sigmas = tf.multiply(tf.exp(-log_sigma), (data_term - model_term))
    return delta_log_sigmas


def error(v0, v1):
    '''
    calculate reconstruction error
    Args:
        v0: array of initial values of visible units, shape: [batchsize,num_vUnits]
        v1: array of values of visible units after running markov chain, shape: [batchsize,num_vUnits]
    Returns:
        reconstruction error (scalar)
    '''
    return tf.reduce_mean(tf.square(v0 - v1))


def sparse_terms(p, q, h_, v, log_sigmas=None):
    '''
    calculate sparsity terms for hbiases and weights
    Args:
        p: goal probability for the hidden units to be on (scalar)
        q: mean activation probabilities of hidden units, shape:[1,num_hUnits]
        h_: activation probabilities of hidden units, shape:[batchsize,num_hUnits]
        v: values of visible units, shape:[batchsize,num_vUnits]
    Returns:
        sparsity terms for hbiases, shape:[1,num_hUnits]
        sparsity terms for weights, shape:[num_vUnits,num_hUnits]
    '''
    sparse_term_hbias = q - p
    if log_sigmas != None:
        v = tf.divide(v, tf.exp(log_sigmas))
    sparse_term_w = tf.matmul(v, (h_ - p), transpose_a=True) / tf.cast(tf.shape(v)[0], tf.float32)
    return sparse_term_hbias, sparse_term_w
