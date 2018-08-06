import numpy as np
import tensorflow as tf
import pickle
from sklearn.decomposition import PCA
import make_datasets as make_ds
from sklearn.manifold import TSNE
import dimensionless_disrcrimination_value as ddv
from RBM import RBM
import train_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
font = {'family': 'sans-serif',
        'weight': 'medium',
        'size': 12}
plt.rc('font', **font)


def make_sample_indices(data, num_samples):
    indices = []
    for _ in range(num_samples):
        ind = np.random.randint(low=0, high=data.shape[0])
        indices.append(ind)
    return indices


def plot_PCA_and_TSNE(data, labels, save_path, save_name, one_hot_labels=True,
                      return_discrimination_value=False):
    pca = PCA(n_components=2)

    pca_samples = data
    pca_labels = labels

    print('Shape PCA input samples: ', pca_samples.shape)

    if one_hot_labels == True:
        classes = np.argmax(pca_labels, axis=1)
    else:
        classes = pca_labels
    print('Shape PCA class labels: ', classes.shape)

    if return_discrimination_value == True:
        print('calculating discrimination value...')
        discrim_value = ddv.discrimination_value(pca_samples, classes, 'z')

    pca_samples_reduced = pca.fit_transform(pca_samples)
    print('Shape reduced PCA samples: ', pca_samples_reduced.shape)

    num_classes = len(np.unique(classes))
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot2grid((1, 1), (0, 0))
    for i in range(num_classes):
        pca_samples_class = pca_samples_reduced[np.where(classes == i)[0]]

        ax.scatter(pca_samples_class[:, 0], pca_samples_class[:, 1], color=cmap[i], label='class {}'.format(i),
                   alpha=0.8)
    ax.legend()
    ax.set_xlabel('PCA component 1 \n discrimination value: {}'.format(discrim_value))
    ax.set_ylabel('PCA component 2')
    ax.set_title('PCA ' + save_name)
    plt.savefig(fname=save_path + save_name + '_PCA.pdf', format='pdf')
    plt.show()

    # TSNE:
    if pca_samples.shape[1] > 50:
        pca = PCA(n_components=50)
        tsne_samples = pca.fit_transform(pca_samples)
    else:
        tsne_samples = pca_samples
    print(tsne_samples.shape)
    tsne = TSNE(n_components=2)
    tsne_samples_reduced = tsne.fit_transform(tsne_samples)
    print('Shape reduced TSNE samples: ', tsne_samples_reduced.shape)
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot2grid((1, 1), (0, 0))
    for i in range(num_classes):
        tsne_samples_class = tsne_samples_reduced[np.where(classes == i)[0]]
        ax.scatter(tsne_samples_class[:, 0], tsne_samples_class[:, 1], color=cmap[i], label='class {}'.format(i),
                   alpha=0.8)
    ax.legend()
    ax.set_title('TSNE ' + save_name)
    ax.set_xlabel('discrimination value: {}'.format(discrim_value))
    plt.savefig(fname=save_path + save_name + '_TSNE.pdf', format='pdf')
    plt.show()

    pickle_out = open(save_path + save_name + '_PCA_samples.pickle', 'wb')
    pickle.dump(pca_samples_reduced, pickle_out)
    pickle_out.close()

    pickle_out = open(save_path + save_name + '_PCA_class_labels.pickle', 'wb')
    pickle.dump(classes, pickle_out)
    pickle_out.close()

    pickle_out = open(save_path + save_name + '_TSNE_samples.pickle', 'wb')
    pickle.dump(tsne_samples_reduced, pickle_out)
    pickle_out.close()

    if return_discrimination_value is True:
        return discrim_value


#######################################################

def layerwise_activations(dbn, data, num_activations, normalize=False):
    """
    return activations of DBN layers
    :param dbn: Deep belief network (Dictionary of RBMs)
    :param data: Data for which activations should be calculated (np array [num_samples,num_features])
    :param num_activations: number of upward propagations over which shall be averaged
    :param normalize: normalize feature activations across all samples to [0,1]
    :return: dictionary of feature activations for each layer
    """
    # initialize dictionary for activations of each layer
    activations = {}
    for layer_index in range(len(dbn)):
        # initialize each layers activations to zeros
        activations['layer_{}'.format(layer_index)] = np.zeros((data.shape[0],
                                                                dbn['layer_{}'.format(layer_index)].num_hunits),
                                                               dtype=np.float32)

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
                    one_epoch_activations.append(sess.run(dbn_activations))
                except tf.errors.OutOfRangeError:
                    break
            for layer_index in range(len(dbn)):
                layer_activations_one_epoch = [act[layer_index] for act in one_epoch_activations]
                layer_activations_one_epoch = np.concatenate(layer_activations_one_epoch, axis=0)
                activations['layer_{}'.format(layer_index)] = np.add(activations['layer_{}'.format(layer_index)],
                                                                     layer_activations_one_epoch)
            one_epoch_activations = []
    tf.reset_default_graph()
    for layer_index in range(len(dbn)):
        activations['layer_{}'.format(layer_index)] = activations['layer_{}'.format(layer_index)] / num_activations
    if normalize is True:
        for layer_index in range(len(dbn)):
            act = activations['layer_{}'.format(layer_index)]
            normalized_act = act / (np.amax(act, axis=0, keepdims=True))
            normalized_act = np.nan_to_num(normalized_act)
            #                 normalized_act = act/(np.amax(act, axis=1).reshape((act.shape[0], 1)))
            activations['layer_{}'.format(layer_index)] = normalized_act
    return activations


#######################################################


def generate_dbn(layer_sizes, layer_types):
    """
    Initialize a DBN
    :param layer_sizes: list of layer sizes including input layer
    :param layer_types: list of layer types (bb, cb, gb or gr)
    :return: initialized DBN (dictionary of RBM layers)
    """
    dbn = {}
    for layer_index in range(len(layer_sizes) - 1):
        dbn['layer_{}'.format(layer_index)] = RBM(layer_sizes[layer_index],
                                                  layer_sizes[layer_index + 1],
                                                  layer_types[layer_index],
                                                  layer_index=layer_index)
    return dbn


def extract_value(dbn_train_params, key, li=0, default_value=None):
    """
    get value from DBN_train_params
    Args:
        dbn_train_params: dictionary of training parameters, dictionary
        key: dictionary key of value to get, string
        li: layer index, integer
        default_value: default value if no value is given in DBN_params
    Returns:
        value
    """
    try:
        value = dbn_train_params[key][li]
    except (KeyError, IndexError):
        print('Key "' + key + '" not defined for layer_index ' + '{}.'.format(li))
        value = default_value
        print('Using default value: ' + '{}.'.format(default_value))
    return value
