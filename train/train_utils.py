import tensorflow as tf
import train.sampling as sampling
import train.make_datasets as make_ds


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
    if get_activations:
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
            # alternative without log_sigmas (works, but does it work better?):
            activation = tf.nn.relu(tf.matmul(batch, tf.constant(layer.weights)) + tf.constant(layer.hbiases))

            # activation = tf.nn.relu(tf.matmul(tf.divide(batch, tf.exp(tf.constant(layer.log_sigmas))),
            #                                   tf.constant(layer.weights))
            #                         + tf.constant(layer.hbiases))
            batch = activation
        else:
            # TODO: add remaining layer type (cr)
            activation = sampling.probs_h_given_v(batch,
                                                  tf.constant(layer.weights),
                                                  tf.constant(layer.hbiases))
            batch = sampling.sampling(activation)
        if get_activations:
            activations.append(activation)
    if get_activations:
        return activations
    else:
        return batch


def get_one_batch(train_data, batch_size):
    dataset = make_ds.simple_dataset(train_data, batch_size, shuffle_buffer=10000, cache=False)
    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        first_batch = sess.run(batch)
    tf.reset_default_graph()
    return first_batch


def calculate_energy(layer_type, visible_states, hidden_states, vbiases, hbiases, weights, log_sigmas=None):
    if layer_type == 'gb' or layer_type == 'gr':
        weights_term = -tf.reduce_sum(tf.multiply(tf.matmul(tf.divide(visible_states, tf.exp(log_sigmas)),
                                                            weights), hidden_states), axis=1)
        vbiases_term = tf.reduce_sum(tf.divide(tf.square(visible_states - vbiases), 2 * tf.exp(log_sigmas)), axis=1)
        hbiases_term = -tf.matmul(hidden_states, hbiases, transpose_b=True)
    else:
        weights_term = -tf.reduce_sum(tf.multiply(tf.matmul(visible_states, weights), hidden_states), axis=1)
        # [[v0w00+v1w10,v0w01+v1w11,...](1)
        # [v0w00+v1w10,v0w01+v1w11,...](2)]
        vbiases_term = -tf.matmul(visible_states, vbiases, transpose_b=True)
        hbiases_term = -tf.matmul(hidden_states, hbiases, transpose_b=True)

    energy = weights_term + tf.reshape(vbiases_term, [-1]) + tf.reshape(hbiases_term, [-1])
    return energy


def get_best_learning_rate(batch, layer_type, etas, cd_steps,
                           train_vbiases, train_hbiases, train_weights,
                           delta_vbiases, delta_hbiases, delta_weights,
                           train_log_sigmas=None, delta_log_sigmas=None):
    approx_likelihoods = []
    v0_ = batch
    h0_, h0, vn_, vn, hn_ = cd_procedure(batch, layer_type, cd_steps,
                                         train_vbiases, train_hbiases, train_weights, train_log_sigmas)
    hn = sampling.sampling(hn_)

    e_model = calculate_energy(layer_type, vn, hn, train_vbiases, train_hbiases, train_weights, train_log_sigmas)

    for candidate_eta in etas:
        vbiases_prime = train_vbiases + candidate_eta * delta_vbiases
        hbiases_prime = train_hbiases + candidate_eta * delta_hbiases
        weights_prime = train_weights + candidate_eta * delta_weights
        if layer_type == 'gb':
            log_sigmas_prime = train_log_sigmas + candidate_eta * delta_log_sigmas
        else:
            log_sigmas_prime = None

        e_prime = calculate_energy(layer_type, v0_, h0, vbiases_prime, hbiases_prime, weights_prime, log_sigmas_prime)
        e_prime_model = calculate_energy(layer_type, vn, hn, vbiases_prime, hbiases_prime, weights_prime,
                                         log_sigmas_prime)
        e_prime = tf.cast(e_prime, tf.float64)
        rho_prime = tf.exp(-e_prime)

        p_prime = tf.reduce_mean(tf.divide(rho_prime,
                                           tf.reduce_mean(tf.exp(tf.cast(-e_prime_model - e_model, tf.float64)))))
        approx_likelihoods.append(p_prime)

    approx_likelihoods = tf.stack(approx_likelihoods, axis=0)
    ind_eta = tf.argmax(approx_likelihoods)

    return ind_eta
