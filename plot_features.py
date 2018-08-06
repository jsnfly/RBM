import numpy as np
import pickle
import matplotlib.pyplot as plt


def visualize_features(dbn, layer_index, feature_index):
    receptive_field = dbn['layer_{}'.format(layer_index)].weights[:, feature_index]
    receptive_field = np.expand_dims(receptive_field, 0)
    for li in range(layer_index - 1, -1, -1):
        print('shape receptive_field: ', receptive_field.shape)
        weights = dbn['layer_{}'.format(li)].weights
        receptive_field = np.matmul(receptive_field, np.transpose(weights))
        receptive_field = receptive_field/np.amax(receptive_field)
    return receptive_field


def load_dbn(file_path):
    pickle_in = open(file_path, 'rb')
    dbn = pickle.load(pickle_in)
    pickle_in.close()
    return dbn


def main():
    dbn_path = '/home/jonas/HDD/DBN_MNIST_runs/Runs_equal_size_NEW/run_0/DBN.pickle'
    dbn = load_dbn(dbn_path)

    # plot features
    save_path = None
    layer_index = 5
    for i in range(10):
        receptive_field = visualize_features(dbn, layer_index, i)
        plt.imshow(receptive_field.reshape([28, 28]), cmap='binary')
        if save_path is not None:
            plt.savefig(save_path + 'receptive_field_layer{}neuron{}.png'.format(layer_index, i),
                        format='png')
        plt.show()


if __name__ == '__main__':
    main()
