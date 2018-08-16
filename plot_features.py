import numpy as np
import tensorflow as tf
import pickle
from helper_functions import downward_propagation
import matplotlib.pyplot as plt
import os
from pathlib import Path


def visualize_features(dbn, layer_index, feature_index):
    receptive_field = dbn['layer_{}'.format(layer_index)].weights[:, feature_index]
    receptive_field = np.expand_dims(receptive_field, 0)
    for li in range(layer_index - 1, -1, -1):
        print('shape receptive_field: ', receptive_field.shape)
        weights = dbn['layer_{}'.format(li)].weights
        receptive_field = np.matmul(receptive_field, np.transpose(weights))
        receptive_field = receptive_field/np.mean(receptive_field)
    return receptive_field


# def calc_grads_wrt_input(dbn):


def load_dbn(file_path):
    pickle_in = open(file_path, 'rb')
    dbn = pickle.load(pickle_in)
    pickle_in.close()
    return dbn


def main():
    tf.reset_default_graph()
    dbn_path = Path('GR_MNIST_sparse1_2/')
    dbn_path = os.fspath(dbn_path / 'dbn.pickle')
    dbn = load_dbn(dbn_path)

    print(dbn['layer_0'].num_hunits)
    # # plot features with receptive field approach
    # save_path = None
    # layer_index = 2
    # for i in range(5):
    #     receptive_field = visualize_features(dbn, layer_index, i)
    #     plt.imshow(receptive_field.reshape([28, 28]), cmap='winter')
    #     if save_path is not None:
    #         plt.savefig(save_path + 'receptive_field_layer{}neuron{}.png'.format(layer_index, i),
    #                     format='png')
    #     plt.show()

    # plot features with downward propagation

    # scale weights
    for li in range(0, len(dbn)):
        # scale_factor = 0.05 / np.mean(np.abs(dbn['layer_{}'.format(li)].weights)) / (li+1)
        layer = dbn['layer_{}'.format(li)]
        print(np.mean(np.abs(layer.weights)))
        # layer.set_weights(layer.weights*scale_factor)
        # print(np.mean(np.abs(layer.weights)))

    layer_index = 0
    num_runs = 1
    layer = dbn['layer_{}'.format(layer_index)]

    activation_placeholder = tf.placeholder(tf.float32)
    input_activation = downward_propagation(activation_placeholder, dbn, layer_index)
    with tf.Session() as sess:
        fig, axes = plt.subplots(4, 4, figsize=(2.895, 2.895))
        inds = np.random.choice(range(layer.num_hunits), size=16, replace=False)
        for c, ax in enumerate(axes.flat):
            i = inds[c]
            activation = np.zeros(shape=[num_runs, layer.num_hunits]).astype(np.float32)
            activation[:, i] = 100.0
            result = sess.run(input_activation, feed_dict={activation_placeholder: activation})
            result = np.mean(result, axis=0)
            im = ax.imshow(result.reshape([28, 28]), cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
        # fig.colorbar(im)
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(os.fspath('GR_MNIST_sparse1_2/features.svg'), format='svg')
        plt.show()
    tf.reset_default_graph()


if __name__ == '__main__':
    main()
