import numpy as np
import tensorflow as tf
import pickle
import train.sampling as sampling
import train.train_utils as train_utils
import train.make_datasets as make_ds
import matplotlib.pyplot as plt
from train.rbm import RBM
from scipy.signal import savgol_filter

font = {'family':   'sans-serif',
        'weight':   'medium',
        'size':     11}
plt.rc('font', **font)


def make_sample_indices(data, num_samples):
    """
    generate random sample indices
    :param data: np.array, shape: [num_samples, num_features]
    :param num_samples: number of indices to be returned
    :return: list of indices
    """
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    return indices


def layerwise_activations(dbn, data, num_activations):
    """
    return activations of DBN layers
    :param dbn: Deep belief network (list of RBMs)
    :param data: Data for which activations should be calculated (np array [num_samples,num_features])
    :param num_activations: number of upward propagations over which shall be averaged
    :return: list of feature activations for each layer
    """
    # initialize dictionary for activations of each layer
    activations = []
    for layer_index in range(len(dbn)):
        # initialize each layers activations to zeros
        activations.append(np.zeros(shape=(data.shape[0], dbn[layer_index].num_hunits), dtype=np.float32))

    # list of lists where each inner list has one activation-batch for each layer
    one_epoch_activations = []

    batch_size = 256
    dataset = make_ds.simple_dataset(data, batch_size, shuffle_buffer=0, cache=True)
    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()

    last_layer_index = len(dbn) - 1
    dbn_activations = train_utils.upward_propagation(batch, dbn, last_layer_index, get_activations=True)

    with tf.Session() as sess:
        for i in range(num_activations):
            sess.run(iterator.initializer)
            while True:
                try:
                    # append list of layer activations for each batch
                    one_epoch_activations.append(sess.run(dbn_activations))
                except tf.errors.OutOfRangeError:
                    break
            for layer_index in range(len(dbn)):
                # extract the activation batches for each layer
                layer_activations_one_epoch = [act[layer_index] for act in one_epoch_activations]
                # concatenate them to get array of activations of the layer with activations for all samples
                layer_activations_one_epoch = np.concatenate(layer_activations_one_epoch, axis=0)
                # add layer activations to activations for all other runs
                activations[layer_index] = np.add(activations[layer_index], layer_activations_one_epoch)
            # reset epoch activations
            one_epoch_activations = []
    tf.reset_default_graph()
    for layer_index in range(len(dbn)):
        # take average of all activation runs for each layer
        activations[layer_index] = activations[layer_index] / num_activations
    return activations


def generate_dbn(layer_sizes, layer_types):
    """
    Initialize a DBN
    :param layer_sizes: list of layer sizes including input layer
    :param layer_types: list of layer types (bb, cb, gb or gr)
    :return: initialized DBN (list of RBM layers)
    """
    dbn = []
    for layer_index in range(len(layer_sizes) - 1):
        layer = RBM(layer_sizes[layer_index],
                    layer_sizes[layer_index + 1],
                    layer_types[layer_index],
                    layer_index=layer_index)
        dbn.append(layer)
    return dbn


def extract_value(dbn_train_params, key, li=0, default_value=None):
    """
    get value from DBN_train_params
    :param dbn_train_params: dictionary of training parameters
    :param key: dictionary key of value to get (string)
    :param li: layer index (integer)
    :param default_value: default value if no value is given in dbn_train_params
    :return: value
    """
    try:
        value = dbn_train_params[key][li]
    except (KeyError, IndexError):
        print('Key "' + key + '" not defined for layer_index ' + '{}.'.format(li))
        value = default_value
        print('Using default value: ' + '{}.'.format(default_value))
    return value


def load_dbn(file_path):
    pickle_in = open(file_path, 'rb')
    dbn = pickle.load(pickle_in)
    pickle_in.close()
    return dbn


def downward_propagation(activation, dbn, layer_index):
    for li in range(layer_index, -1, -1):
        layer = dbn[li]
        if layer.layer_type == 'gr' or layer.layer_type == 'gb':
            activation = tf.matmul(activation, tf.constant(layer.weights), transpose_b=True) + tf.constant(
                layer.vbiases)
        elif layer.layer_type == 'bb':
            activation = sampling.sampling(sampling.probs_v_given_h(activation,
                                                                    tf.constant(layer.weights),
                                                                    tf.constant(layer.vbiases)))
        else:
            # TODO: implement other layer types
            raise TypeError('Layer type not implemented yet')
    return activation


def downward_propagate_features(dbn, layer_index, feature_indices, num_runs=1, num_neurons=1, activation_value=1.0):
    """
    propagate features downward to get their receptive input fields
    :param dbn: DBN (list of RBM layers)
    :param layer_index: layer index of a given neuron
    :param feature_indices: indices of neurons in a given layer (list of integers)
    :param num_runs: number of times the features is downward propagated and averaged over in the end
    :param num_neurons: number of neurons which are set to activation_value
    :param activation_value: to which value to set the feature activation (float)
    :return: list of receptive input fields
    """
    receptive_fields = []
    layer = dbn[layer_index]
    # create TF graph
    activation_placeholder = tf.placeholder(tf.float32)
    input_activation = downward_propagation(activation_placeholder, dbn, layer_index)
    with tf.Session() as sess:
        for i in feature_indices:
            activation = np.zeros(shape=[num_runs, layer.num_hunits]).astype(np.float32)
            activation[:, i:i+num_neurons] = activation_value
            result = sess.run(input_activation, feed_dict={activation_placeholder: activation})
            result = np.mean(result, axis=0)
            receptive_fields.append(result)
    tf.reset_default_graph()
    return receptive_fields


def plot_class_activations(save_path, dbn, data, labels, num_activations, one_hot=True):

    if one_hot is True:
        classes = np.argmax(labels, axis=1)
    else:
        classes = labels

    num_classes = np.amax(classes)+1

    fig, axes = plt.subplots(1, len(dbn), figsize=(5.79, 3.79))
    for c in range(num_classes):
        cls_indices = np.where(classes == c)[0]
        cls_samples = data[cls_indices]
        cls_activations = layerwise_activations(dbn, cls_samples, num_activations)
        for li, layer_act in enumerate(cls_activations):
            smooth_act = savgol_filter(np.mean(layer_act, axis=0), 15, 3)
            axes.flat[li].plot(smooth_act, label='class {}'.format(c), alpha=1.0)

    for li,ax in enumerate(axes.flat):
        ax.set_title('Layer {}'.format(li))
        ax.legend()
    plt.savefig(save_path + 'layerwise_activations.png', format='png')
    plt.show()
