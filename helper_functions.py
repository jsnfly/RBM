import numpy as np
import tensorflow as tf
import pickle
import sampling
import train_utils
import make_datasets as make_ds
from RBM import RBM
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


def plot_pca_or_tsne(samples, labels, one_hot_labels=True):
    """
    generates plot for PCA or TSNE samples with two components
    :param samples: PCA or TSNE samples, shape: [num_samples, 2]
    :param labels: either one hot labels or class indices
    :param one_hot_labels: whether one hot labels are given
    :return: figure with scatter plot of PCA/TSNE samples
    """
    # create class labels from one-hots if necessary
    if one_hot_labels is True:
        classes = np.argmax(labels, axis=1)
    else:
        classes = labels

    # PCA
    num_classes = len(np.unique(classes))
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
    fig = plt.figure(figsize=(5.79, 4))
    ax = plt.subplot2grid((1, 1), (0, 0))
    for i in range(num_classes):
        cls_indices = np.where(classes == i)[0]
        samples_class = samples[cls_indices]
        ax.scatter(samples_class[:, 0], samples_class[:, 1], color=cmap[i], label='class {}'.format(i), alpha=0.8)
    ax.legend()
    return fig


def calculate_pca(data):
    """
    calculate PCA
    :param data: np.array, shape: [num_samples, num_features]
    :return: PCA samples
    """
    # define PCA
    pca = PCA(n_components=2)

    # apply pca
    pca_samples_reduced = pca.fit_transform(data)

    return pca_samples_reduced


def calculate_tsne(data):
    """
    calculate TSNE for given data and labels (if dimension is greater thant 50, the TSNE is calculated by
    first reducing the dimension to 50 using PCA)
    :param data: np.array, shape: [num_samples, num_features]
    :return: TSNE samples
    """
    # reduce dimensions using PCA if necessary
    if data.shape[1] > 50:
        pca = PCA(n_components=50)
        data = pca.fit_transform(data)
    else:
        pass

    tsne = TSNE(n_components=2)
    tsne_samples_reduced = tsne.fit_transform(data)

    return tsne_samples_reduced


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
