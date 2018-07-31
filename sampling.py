import tensorflow as tf


def probs_h_given_v(v, W, hbias):
    '''
    calculate probabilities for hidden units to be on given the visible units
    Args:
        v: values of visible units, shape: [batch_size,num_vUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
        hbias: biases of hidden units, shape: [1,num_hUnits]
    Returns:
        activation probabilities of hidden units, shape:[batch_size,num_hUnits]
    '''
    z_i = tf.matmul(v, W) + hbias
    return tf.nn.sigmoid(z_i)


def probs_v_given_h(h, W, vbias):
    '''
    calculate probabilities for visible units to be on given the hidden units
    Args:
        h: values of hidden units, shape: [batch_size,num_hUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
        vbias: biases of visible units, shape: [1,num_vUnits]
    Returns:
        activation probabilities of visible units, shape:[batch_size,num_vUnits]
    '''
    z_i = tf.matmul(h, W, transpose_b=True) + vbias
    return tf.nn.sigmoid(z_i)


def sampling(probs):
    '''
    sample from probabilities of hidden units to be on
    Args:
        probs: activation probabilities, shape:[batch_size]
    Returns:
        samples: sampled binary values, shape:[num_batches,num_Units]
    '''
    samples = tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs), dtype=tf.float32)))
    return samples


def v_cont_given_h(h, W, vbias, epsilon=1.e-14):
    '''
    calculate values of visible units for truncated exponential units
    Args:
        h: values of hidden units, shape: [batch_size,num_hUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
        vbias: biases of visible units, shape: [1,num_vUnits]
        epsilon: small constant to prevent division by zero, float
    Returns:
        continuous values in [0,1] of visible units, shape:[batch_size,num_vUnits]
    '''
    z_i = tf.matmul(h, W, transpose_b=True) + vbias
    z_i = tf.cast(z_i, tf.float64) + tf.cast(epsilon, tf.float64)
    samples = tf.log(1 - tf.random_uniform(tf.shape(z_i), dtype=tf.float64) * (1 - tf.exp(z_i))) / z_i
    return tf.cast(samples, tf.float32)


def v_gauss_given_h(h, W, vbias, log_sigma):
    '''
    calculate values of visible units for gaussian visible units
    Args:
        h: values of hidden units, shape: [batch_size,num_hUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
        vbias: biases of visible units, shape: [1,num_vUnits]
        log_sigma: log of squared standard deviations of visible units, shape: [1,num_vUnits]
    Returns:
        continuous values in [0,1] of hidden units, shape:[batch_size,num_hUnits]
    '''
    z_i = tf.matmul(h, W, transpose_b=True) + vbias
    gaussian = tf.distributions.Normal(loc=z_i, scale=tf.exp(log_sigma))
    samples = tf.reshape(gaussian.sample([1]), shape=tf.shape(z_i))
    return samples


def probs_h_given_v_gauss(v, W, hbias, log_sigma):
    '''
    calculate probabilities for hidden units to be on given gaussian visible units
    Args:
        v: values of visible units, shape: [batch_size,num_vUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
        hbias: biases of hidden units, shape: [1,num_hUnits]
        log_sigma: log of squared standard deviations of visible units, shape: [1,num_vUnits]
    Returns:
        activation probabilities of hidden units, shape:[batch_size,num_hUnits]
    '''
    z_i = tf.matmul(tf.divide(v, tf.exp(log_sigma)), W) + hbias
    return tf.nn.sigmoid(z_i)


def ReLU_h_given_v_gauss(v, W, hbias, log_sigma):
    '''
    calculate probabilities for hidden units to be on given gaussian visible units
    Args:
        v: values of visible units, shape: [batch_size,num_vUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
        hbias: biases of hidden units, shape: [1,num_hUnits]
    Returns:
        activation of ReLU hidden units, shape:[batch_size,num_hUnits]
    '''
    z_i = tf.matmul(tf.divide(v, tf.exp(log_sigma)), W) + hbias
    gaussian = tf.distributions.Normal(loc=z_i, scale=tf.nn.sigmoid(z_i))
    samples = tf.reshape(gaussian.sample([1]), shape=tf.shape(z_i))
    return tf.nn.relu(samples)


def ReLU_h_given_v_cont(v, W, hbias):
    '''
    calculate probabilities for hidden units to be on given gaussian visible units
    Args:
        v: values of visible units, shape: [batch_size,num_vUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
        hbias: biases of hidden units, shape: [1,num_hUnits]
    Returns:
        activation of ReLU hidden units, shape:[batch_size,num_hUnits]
    '''
    z_i = tf.matmul(v, W) + hbias
    gaussian = tf.distributions.Normal(loc=z_i, scale=tf.nn.sigmoid(z_i))
    samples = tf.reshape(gaussian.sample([1]), shape=tf.shape(z_i))
    return tf.nn.relu(samples)
