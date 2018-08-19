import numpy as np
import tensorflow as tf
import pickle
from helper_functions import *
import matplotlib.pyplot as plt
import os
from pathlib import Path


def get_receptive_field(dbn, layer_index, feature_index):
    """
    calculate the receptive input field of a given neuron by weighing the contribution from each
    lower layer by the weights connecting them and normalizing after each step to prevent vanishing/exploding gradients
    :param dbn: DBN (list of RBM layers)
    :param layer_index: layer index of a given neuron
    :param feature_index: index of the neuron in a given layer
    :return: receptive input field
    """
    receptive_field = dbn[layer_index].weights[:, feature_index]
    receptive_field = np.expand_dims(receptive_field, 0)
    for li in range(layer_index - 1, -1, -1):
        weights = dbn[li].weights
        receptive_field = np.matmul(receptive_field, np.transpose(weights))
        receptive_field = receptive_field/np.mean(receptive_field)
    return receptive_field


# def calc_grads_wrt_input(dbn):


def downward_propagate_features(dbn, layer_index, feature_indices, num_runs=1, activation_value=1.0):
    """
    propagate features downward to get their receptive input fields
    :param dbn: DBN (list of RBM layers)
    :param layer_index: layer index of a given neuron
    :param feature_indices: indices of neurons in a given layer (list of integers)
    :param num_runs: number of times the features is downward propagated and averaged over in the end
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
            activation[:, i:i+25] = activation_value
            result = sess.run(input_activation, feed_dict={activation_placeholder: activation})
            result = np.mean(result, axis=0)
            receptive_fields.append(result)
    tf.reset_default_graph()
    return receptive_fields


def main():
    dbn_path = Path('Test2/')
    dbn_path = os.fspath(dbn_path / 'dbn.pickle')
    dbn = load_dbn(dbn_path)

    save_path = 'Test2/'
    layer_index = 7
    layer = dbn[layer_index]
    inds = np.random.choice(range(layer.num_hunits), size=16, replace=False)

    # # plot features with receptive field approach
    # fig, axes = plt.subplots(4, 4, figsize=(2.895, 2.895))
    # for c, ax in enumerate(axes.flat):
    #     i = inds[c]
    #     receptive_field = get_receptive_field(dbn, layer_index, i)
    #     ax.imshow(receptive_field.reshape([28, 28]), cmap='seismic')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # if save_path is not None:
    #     plt.savefig(save_path + 'receptive_fields_layer{}_v1.png'.format(layer_index), format='png')
    # plt.show()

    # plot features with downward propagate approach
    receptive_fields = downward_propagate_features(dbn, layer_index, inds, num_runs=1000, activation_value=1.0)
    fig, axes = plt.subplots(4, 4, figsize=(2.895, 2.895))
    for c, ax in enumerate(axes.flat):
        receptive_field = receptive_fields[c]
        ax.imshow(receptive_field.reshape([28, 28]), cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save_path is not None:
        plt.savefig(save_path + 'receptive_fields_layer{}_v2.png'.format(layer_index), format='png')
    plt.show()


if __name__ == '__main__':
    main()
