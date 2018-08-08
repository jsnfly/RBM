import numpy as np
import tensorflow as tf
import pickle
from helper_functions import downward_propagation
import matplotlib.pyplot as plt


def visualize_features(dbn, layer_index, feature_index):
    receptive_field = dbn['layer_{}'.format(layer_index)].weights[:, feature_index]
    receptive_field = np.expand_dims(receptive_field, 0)
    for li in range(layer_index - 1, -1, -1):
        print('shape receptive_field: ', receptive_field.shape)
        weights = dbn['layer_{}'.format(li)].weights
        receptive_field = np.matmul(receptive_field, np.transpose(weights))
        receptive_field = receptive_field/np.mean(receptive_field)
    return receptive_field


def load_dbn(file_path):
    pickle_in = open(file_path, 'rb')
    dbn = pickle.load(pickle_in)
    pickle_in.close()
    return dbn


def main():
    dbn_path = '/home/jonas/HDD/DBN_MNIST_runs/Runs_equal_size_NEW/run_8/DBN.pickle'
    dbn = load_dbn(dbn_path)

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
    layer_index = 28
    num_runs = 400
    layer = dbn['layer_{}'.format(layer_index)]

    activation_placeholder = tf.placeholder(tf.float32)
    input_activation = downward_propagation(activation_placeholder, dbn, layer_index)

    with tf.Session() as sess:
        fig, axes = plt.subplots(4, 4)
        # fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for ax in axes.flat:
            i = np.random.choice(range(layer.num_hunits), replace=False)
            activation = np.zeros(shape=[1, layer.num_hunits]).astype(np.float32)
            activation[0, i] = 1.0
            result = np.zeros((1, dbn['layer_0'].num_vunits))
            for r in range(num_runs):
                res = sess.run(input_activation, feed_dict={activation_placeholder: activation})
                result = np.add(result, res)
            result = result/num_runs
            ax.imshow(result.reshape([28, 28]), cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    tf.reset_default_graph()


if __name__ == '__main__':
    main()
