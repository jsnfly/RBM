import tensorflow as tf
import train.sampling as sampling


# contrastive divergence:
def cd_step(layer_type, v, vbiases, hbiases, weights, log_sigmas=None):
    """
    
    :param layer_type: RBM layer type ('bb', 'cb', 'gb', 'gr' or 'cr')
    :param v: values of visible units, shape: [batch_size, num_vunits]
    :param vbiases: biases of visible units, shape: [1, num_vunits]
    :param hbiases: biases of hidden units, shape: [1, num_hunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param log_sigmas: logarithmic squared standard deviation (only when using Gaussian units), shape: [1, num_vunits]
    :return: 
        h0_: activation(-probabilities) of hidden units after one Gibbs step
        h0: (sampled) activation of hidden units after one Gibbs step
        vn_: activation(-probabilities) of visible units after two Gibbs steps
        vn: (sampled) activation of visible units after two Gibbs step
        hn_: activation(-probabilities) of hidden units after two Gibbs Steps
    """
    if layer_type == 'bb':
        h0_ = sampling.probs_h_given_v(v, weights, hbiases)
        h0 = sampling.sampling(h0_)
        vn_ = sampling.probs_v_given_h(h0, weights, vbiases)
        vn = sampling.sampling(vn_)
        hn_ = sampling.probs_h_given_v(vn, weights, hbiases)

    elif layer_type == 'cb':
        h0_ = sampling.probs_h_given_v(v, weights, hbiases)
        h0 = sampling.sampling(h0_)
        vn_ = sampling.v_cont_given_h(h0, weights, vbiases)
        vn = vn_
        hn_ = sampling.probs_h_given_v(vn, weights, hbiases)

    elif layer_type == 'gb':
        h0_ = sampling.probs_h_given_v_gauss(v, weights, hbiases, log_sigmas)
        h0 = sampling.sampling(h0_)
        vn_ = sampling.v_gauss_given_h(h0, weights, vbiases, log_sigmas)
        vn = vn_
        hn_ = sampling.probs_h_given_v_gauss(vn, weights, hbiases, log_sigmas)

    elif layer_type == 'gr':
        h0_ = sampling.relu_h_given_v_gauss(v, weights, hbiases, log_sigmas)
        h0 = h0_
        vn_ = sampling.v_gauss_given_h(h0, weights, vbiases, log_sigmas)
        vn = vn_
        hn_ = sampling.relu_h_given_v_gauss(vn, weights, hbiases, log_sigmas)

    elif layer_type == 'cr':
        h0_ = sampling.relu_h_given_v(v, weights, hbiases)
        h0 = h0_
        vn_ = sampling.v_cont_given_h(h0, weights, vbiases)
        vn = vn_
        hn_ = sampling.relu_h_given_v(vn, weights, hbiases)
    else:
        raise TypeError('Given Layer Type is not supported')
    return h0_, h0, vn_, vn, hn_


def cd_procedure(batch, layer_type, cd_steps, vbiases, hbiases, weights, log_sigmas=None):
    """
    perform the CD procedure
    :param batch: one batch of samples, shape: [batch_size, num_vunits]
    :param layer_type: layer_type: RBM layer type ('bb', 'cb', 'gb', 'gr' or 'cr')
    :param cd_steps: number of CD steps to be performed
    :param vbiases: biases of visible units, shape: [1, num_vunits]
    :param hbiases: biases of hidden units, shape: [1, num_hunits]
    :param weights: weight matrix, shape: [num_vunits, num_hunits]
    :param log_sigmas: logarithmic squared standard deviation (only when using Gaussian units), shape: [1, num_vunits]
    :return:
        h0_: activation(-probabilities) of hidden units after one Gibbs step
        h0: (sampled) activation of hidden units after one Gibbs step
        vn_: activation(-probabilities) of visible units after running the Markov chain
        vn: (sampled) activation of visible units after running the Markov chain
        hn_: activation(-probabilities) of hidden units after running the Markov chain
    """
    v0_ = batch
    h0_, h0, vn_, vn, hn_ = cd_step(layer_type, v0_, vbiases, hbiases, weights, log_sigmas)
    for n in range(cd_steps - 1):
        _, _, vn_, vn, hn_ = cd_step(layer_type, vn, vbiases, hbiases, weights, log_sigmas)
    return h0_, h0, vn_, vn, hn_


# other functions used for training:
def upward_propagation(batch, dbn, layer_index, get_activations=False):
    """
    propagate one batch upward through a dbn
    :param batch: one batch of samples, shape: [batch_size, num_vunits]
    :param dbn: the deep belief network (list of RBM instances)
    :param layer_index: index of the layer up to which the batch is propagated (integer)
    :param get_activations: whether to return the activations of all the layers up to (including)
        the layer specified by layer_index
    :return: either the upward propagated batch or a list of activations for each layer
    """
    if get_activations is True:
        activations = []
        num_layers = layer_index + 1
    else:
        num_layers = layer_index
    for li in range(num_layers):
        layer = dbn[li]

        if layer.layer_type == 'gb':
            activation = sampling.probs_h_given_v_gauss(batch,
                                                        tf.constant(layer.weights),
                                                        tf.constant(layer.hbiases),
                                                        tf.constant(layer.log_sigmas))
            batch = sampling.sampling(activation)

        elif layer.layer_type == 'gr':
            # TODO: Check this:
            # # alternative without log_sigmas (works, but does it work better?):
            # activation = tf.nn.relu(tf.matmul(batch, tf.constant(layer.weights)) + tf.constant(layer.hbiases))

            activation = tf.nn.relu(tf.matmul(tf.divide(batch, tf.exp(tf.constant(layer.log_sigmas))),
                                              tf.constant(layer.weights))
                                    + tf.constant(layer.hbiases))
            batch = activation
        else:
            # TODO: add remaining layer type (cr)
            activation = sampling.probs_h_given_v(batch,
                                                  tf.constant(layer.weights),
                                                  tf.constant(layer.hbiases))
            batch = sampling.sampling(activation)
        if get_activations is True:
            activations.append(activation)
    if get_activations is True:
        return activations
    else:
        return batch
