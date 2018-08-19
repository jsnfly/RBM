import tensorflow as tf


def probs_h_given_v(v, weights, hbiases):
    """
    calculate probabilities for binary hidden units to be on, given the visible units
    :param v: values of visible units, shape: [batch_size,num_vunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param hbiases: biases of hidden units, shape: [1, num_hunits]
    :return: activation probabilities of hidden units, shape:[batch_size, num_hunits]
    """
    z_i = tf.matmul(v, weights) + hbiases
    return tf.nn.sigmoid(z_i)


def probs_v_given_h(h, weights, vbiases):
    """
    calculate probabilities for binary visible units to be on given the hidden units
    :param h: values of hidden units, shape: [batch_size, num_hunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param vbiases: biases of visible units, shape: [1, num_vunits]
    :return: activation probabilities of visible units, shape:[batch_size, num_vunits]
    """
    z_i = tf.matmul(h, weights, transpose_b=True) + vbiases
    return tf.nn.sigmoid(z_i)


def sampling(probs):
    """
    sample from activation probabilities of binary visible or hidden units
    :param probs: activation probabilities, shape:[batch_size, num_units]
    :return: sampled binary values, shape:[num_batches, num_units]
    """
    samples = tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs), dtype=tf.float32)))
    return samples


def v_cont_given_h(h, weights, vbiases, epsilon=1.e-14):
    """
    calculate values of visible units for truncated exponential units
    :param h: values of hidden units, shape: [batch_size, num_hunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param vbiases: biases of visible units, shape: [1, num_vunits]
    :param epsilon: small constant to prevent division by zero, float
    :return: continuous values in [0,1] for visible units, shape:[batch_size, num_vunits]
    """
    z_i = tf.matmul(h, weights, transpose_b=True) + vbiases
    z_i = tf.cast(z_i, tf.float64) + tf.cast(epsilon, tf.float64)
    samples = tf.log(1 - tf.random_uniform(tf.shape(z_i), dtype=tf.float64) * (1 - tf.exp(z_i))) / z_i
    return tf.cast(samples, tf.float32)


def v_gauss_given_h(h, weights, vbiases, log_sigmas):
    """
    calculate values of visible units for gaussian units
    :param h: values of hidden units, shape: [batch_size, num_hunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param vbiases: biases of visible units, shape: [1, num_vunits]
    :param log_sigmas: logarithmic squared standard deviation, shape: [1, num_vunits]
    :return: continuous values in [0,1] for visible units, shape:[batch_size, num_vunits]
    """
    z_i = tf.matmul(h, weights, transpose_b=True) + vbiases
    gaussian = tf.distributions.Normal(loc=z_i, scale=tf.exp(log_sigmas))
    samples = tf.reshape(gaussian.sample([1]), shape=tf.shape(z_i))
    return samples


def probs_h_given_v_gauss(v, weights, hbiases, log_sigmas):
    """
    calculate probabilities for binary hidden units to be on, given gaussian visible units
    :param v: values of visible units, shape: [batch_size, num_vunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param hbiases: biases of hidden units, shape: [1, num_hunits]
    :param log_sigmas: logarithmic squared standard deviation, shape: [1, num_vunits]
    :return: activation probabilities of hidden units, shape:[batch_size, num_hunits]
    """
    z_i = tf.matmul(tf.divide(v, tf.exp(log_sigmas)), weights) + hbiases
    return tf.nn.sigmoid(z_i)


def relu_h_given_v_gauss(v, weights, hbiases, log_sigmas):
    """
    calculate values of Rectified linear hidden units, given Gaussian visible units
    :param v: values of visible units, shape: [batch_size, num_vunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param hbiases: biases of hidden units, shape: [1, num_hunits]
    :param log_sigmas: logarithmic squared standard deviation, shape: [1, num_vunits]
    :return: activations of ReLUs, shape: [batch_size, num_hunits]
    """
    z_i = tf.matmul(tf.divide(v, tf.exp(log_sigmas)), weights) + hbiases
    gaussian = tf.distributions.Normal(loc=z_i, scale=tf.nn.sigmoid(z_i))
    samples = tf.reshape(gaussian.sample([1]), shape=tf.shape(z_i))
    return tf.nn.relu(samples)


def relu_h_given_v(v, weights, hbiases):
    """
    calculate values of Rectified linear hidden units, given truncated exponential (or binary) visible units
    :param v: values of visible units, shape: [batch_size, num_vunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param hbiases: biases of hidden units, shape: [1, num_hunits]
    :return: activations of ReLUs, shape: [batch_size, num_hunits]
    """
    z_i = tf.matmul(v, weights) + hbiases
    gaussian = tf.distributions.Normal(loc=z_i, scale=tf.nn.sigmoid(z_i))
    samples = tf.reshape(gaussian.sample([1]), shape=tf.shape(z_i))
    return tf.nn.relu(samples)
