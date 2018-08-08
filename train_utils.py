import tensorflow as tf

import make_datasets as make_ds
import sampling


###################################################################
# contrastive divergence:
def CD_step(layer_type, v0, vbias, hbias, W, log_sigma=None):
    '''
    perform one CD-step
    Args:
        layer_type: type of layer, string
        v0: array of initial values of visible units, shape: [batch_size,num_vUnits]
        vbias: biases of visible units, shape: [1,num_vUnits]
        hbias: biases of hidden units, shape: [1,num_hUnits]
        W: weight matrix, shape: [num_vUnits,num_hUnits]
    Returns:
        h0_: activation probabilities of hidden units (values for continuous rbm), shape:[batch_size,num_hUnits]
        h0: sampled values of hidden units, shape:[batch_size,num_hUnits]
        vn_: activation probabilities of visible units (values for continuous rbm) after one CD step, shape:[batch_size,num_vUnits]
        vn: sampled values of visiblen units after one CD step, shape:[batch_size,num_vUnits]
        hn_: activation probabilities of hidden units (values for continuous rbm) after one CD step, shape:[batch_size,num_hUnits]
    '''
    if layer_type == 'bb':
        h0_ = sampling.probs_h_given_v(v0, W, hbias)
        h0 = sampling.sampling(h0_)
        vn_ = sampling.probs_v_given_h(h0, W, vbias)
        vn = sampling.sampling(vn_)
        hn_ = sampling.probs_h_given_v(vn, W, hbias)

    if layer_type == 'cb':
        h0_ = sampling.probs_h_given_v(v0, W, hbias)
        h0 = sampling.sampling(h0_)
        vn_ = sampling.v_cont_given_h(h0, W, vbias)
        vn = vn_
        hn_ = sampling.probs_h_given_v(vn, W, hbias)

    if layer_type == 'gb':
        h0_ = sampling.probs_h_given_v_gauss(v0, W, hbias, log_sigma)
        h0 = sampling.sampling(h0_)
        vn_ = sampling.v_gauss_given_h(h0, W, vbias, log_sigma)
        vn = vn_
        hn_ = sampling.probs_h_given_v_gauss(vn, W, hbias, log_sigma)

    if layer_type == 'gr':
        h0_ = sampling.ReLU_h_given_v_gauss(v0, W, hbias, log_sigma)
        h0 = h0_
        vn_ = sampling.v_gauss_given_h(h0, W, vbias, log_sigma)
        vn = vn_
        hn_ = sampling.ReLU_h_given_v_gauss(vn, W, hbias, log_sigma)

    if layer_type == 'cr':
        h0_ = sampling.ReLU_h_given_v_cont(v0, W, hbias)
        h0 = h0_
        vn_ = sampling.v_cont_given_h(h0, W, vbias)
        vn = vn_
        hn_ = sampling.ReLU_h_given_v_cont(vn, W, hbias)
    return h0_, h0, vn_, vn, hn_


def CD_procedure(batch, layer_type, CD_steps, train_vbiases, train_hbiases, train_weights, train_log_sigmas=None):
    v0_ = batch
    h0_, h0, vn_, vn, hn_ = CD_step(layer_type, v0_,
                                    train_vbiases, train_hbiases, train_weights, train_log_sigmas)
    for n in range(CD_steps - 1):
        _, _, vn_, vn, hn_ = CD_step(layer_type, vn, train_vbiases, train_hbiases, train_weights, train_log_sigmas)
    return h0_, h0, vn_, vn, hn_


###################################################################

###################################################################
# for adaptive learning rate:
def get_one_batch(train_data, batch_size):
    dataset = make_ds.simple_dataset(train_data, batch_size, shuffle=True, cache=False)
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


def get_best_learning_rate(batch, layer_type, etas, CD_steps,
                           train_vbiases, train_hbiases, train_weights,
                           delta_vbiases, delta_hbiases, delta_weights,
                           train_log_sigmas=None, delta_log_sigmas=None):
    approx_likelihoods = []
    v0_ = batch
    h0_, h0, vn_, vn, hn_ = CD_procedure(batch, layer_type, CD_steps,
                                         train_vbiases, train_hbiases, train_weights, train_log_sigmas)
    hn = sampling.sampling(hn_)

    E_model = calculate_energy(layer_type, vn, hn, train_vbiases, train_hbiases, train_weights, train_log_sigmas)

    for n in range(3):
        candidate_eta = etas[n]
        vbiases_prime = train_vbiases + candidate_eta * delta_vbiases
        hbiases_prime = train_hbiases + candidate_eta * delta_hbiases
        weights_prime = train_weights + candidate_eta * delta_weights
        if layer_type == 'gb':
            log_sigmas_prime = train_log_sigmas + candidate_eta * delta_log_sigmas
        else:
            log_sigmas_prime = None

        E_prime = calculate_energy(layer_type, v0_, h0, vbiases_prime, hbiases_prime, weights_prime, log_sigmas_prime)
        E_prime_model = calculate_energy(layer_type, vn, hn, vbiases_prime, hbiases_prime, weights_prime,
                                         log_sigmas_prime)
        E_prime = tf.cast(E_prime, tf.float64)
        rho_prime = tf.exp(-E_prime)

        P_prime = tf.reduce_mean(tf.divide(rho_prime,
                                           tf.reduce_mean(tf.exp(tf.cast(-E_prime_model - E_model, tf.float64)))))
        approx_likelihoods.append(P_prime)

    approx_likelihoods = tf.stack(approx_likelihoods, axis=0)
    ind_eta = tf.argmax(approx_likelihoods)

    return ind_eta


###################################################################


###################################################################
# other functions used for training:
def upward_propagation(batch, DBN, layer_index, get_activations=False):
    if get_activations == True:
        activations = []
        num_layers = layer_index + 1
    else:
        num_layers = layer_index
    for li in range(num_layers):
        layer = DBN['layer_{}'.format(li)]
        if layer.layer_type == 'gb':
            activation = sampling.probs_h_given_v_gauss(batch,
                                                        tf.constant(layer.weights),
                                                        tf.constant(layer.hbiases),
                                                        tf.constant(layer.log_sigmas))
            batch = sampling.sampling(activation)
        elif layer.layer_type == 'gr':
            activation = tf.nn.relu(tf.matmul(batch, tf.constant(layer.weights)) + tf.constant(layer.hbiases))
            #             activation = sampling.ReLU_h_given_v_gauss(batch,
            #                                                        tf.constant(layer.weights),
            #                                                        tf.constant(layer.hbiases),
            #                                                        tf.constant(layer.log_sigmas))
            batch = activation
            if DBN['layer_{}'.format(li + 1)].layer_type == 'cb':
                batch = tf.nn.sigmoid(batch)
        else:
            activation = sampling.probs_h_given_v(batch,
                                                  tf.constant(layer.weights),
                                                  tf.constant(layer.hbiases))
            batch = sampling.sampling(activation)
        if get_activations == True:
            activations.append(activation)
    if get_activations == True:
        return activations
    else:
        return batch
###################################################################
