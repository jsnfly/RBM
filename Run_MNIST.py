import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from RBM import RBM
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_mnist_data():
    """
    Download mnist data if it does not already exist in directory MNIST_Data, else load it from directory.
    """
    try:
        x_train = np.load('MNIST_Data/train_data.npy')
        y_train = np.load('MNIST_Data/train_labels.npy')
        x_test = np.load('MNIST_Data/test_data.npy')
        y_test = np.load('MNIST_Data/test_labels.npy')

        print('MNIST/Data already exists, loaded files.')

    except IOError:
        if not os.path.exists('MNIST_Data'):
            os.makedirs('MNIST_Data')
        print('Downloading MNIST data.')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

        print('saving MNIST data in MNIST_data/')
        np.save('MNIST_Data/train_data', x_train)
        np.save('MNIST_Data/train_labels', y_train)
        np.save('MNIST_Data/test_data', x_test)
        np.save('MNIST_Data/test_labels', y_test)

    # Reshape data into shape [num_samples,num_features]
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    print('shape training examples: ', x_train.shape)
    print('shape training labels: ', y_train.shape)
    print('shape test examples: ', x_test.shape)
    print('shape test labels: ', y_test.shape)

    return (x_train, y_train), (x_test, y_test)


def treshold_data(data, threshold):
    """
    threshold data with threshold value
    :param data: numpy array
    :param threshold: float
    :return: thresholded data
    """
    data = np.where(data >= threshold, np.ones_like(data), np.zeros_like(data))
    data = data.astype(np.float32)
    return data


def main():
    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    # # threshold data
    # x_train = treshold_data(x_train, 0.5 * 255)

    # plot one random image to check if the format is correct
    plt.imshow(x_train[np.random.randint(x_train.shape[0]), :].reshape([28, 28]), cmap='binary')
    plt.show()

    # Set up Deep Belief Network
    layer_sizes = [784, 512, 32, 16]
    layer_types = ['gr', 'gr', 'gr']
    mnist_dbn = generate_dbn(layer_sizes, layer_types)

    # Training parameters
    dbn_train_params = {
        'epochs': [20, 20, 20],  # number of training epochs
        'batch_size': [128, 128, 128],  # size of one training batch
        'cd_steps': [1, 1, 1],  # number of CD training steps
        'update_vbiases': [True, True, True],  # if false vbiases are set to zero throughout the training
        'learning_rate': [0.005, 0.005, 0.005],  # learning rate at the begin of training
        'lr_decay': [(3, 0.55), (3, 0.55), (3, 0.55)],  # decay of learning rate (every epochs,decay factor)
        'summary_frequency': [250, 250, 250],  # write to summary every x batches
        'sparsity_rate': [0.2, 0.01, 0.01],  # rate with which sparsity is enforced
        'sparsity_goal': [0.2, 0.10, 0.10]  # goal activation probability
    }

    # Train DBN
    summaries_path = '/home/jonas/PycharmProjects/RBM/MNIST_Summaries/Test1/'

    for li in range(len(mnist_dbn)):
        # set up layer summary path
        summary_path = summaries_path + 'layer_{}'.format(li)

        # get training parameters for each layer from DBN_train_params
        layer = mnist_dbn['layer_{}'.format(li)]
        epochs = extract_value(dbn_train_params, 'epochs', li, 10)
        batch_size = extract_value(dbn_train_params, 'batch_size', li, 32)
        cd_steps = extract_value(dbn_train_params, 'cd_steps', li, 1)
        update_vbiases = extract_value(dbn_train_params, 'update_vbiases', li, False)
        start_learning_rate = extract_value(dbn_train_params, 'learning_rate', li, 0.1)
        lr_decay = extract_value(dbn_train_params, 'lr_decay', li, (10, 0.9))
        summary_frequency = extract_value(dbn_train_params, 'summary_frequency', li, 10)
        sparsity_rate = extract_value(dbn_train_params, 'sparsity_rate', li, 0)
        sparsity_goal = extract_value(dbn_train_params, 'sparsity_goal', li, 0)

        layer.train_rbm(train_data=(x_train/255).astype(np.float32),
                        epochs=epochs,
                        batch_size=batch_size,
                        summary_path=summary_path,
                        summary_frequency=summary_frequency,
                        dbn=mnist_dbn,
                        update_vbiases=update_vbiases,
                        start_learning_rate=start_learning_rate,
                        learning_rate_decay=lr_decay,
                        cd_steps=cd_steps,
                        sparsity_rate=sparsity_rate,
                        sparsity_goal=sparsity_goal)

        # save trained dbn:
        pickle_out = open(summaries_path + 'dbn.pickle', 'wb')
        pickle.dump(mnist_dbn, pickle_out)
        pickle_out.close()

        # save parameter dicts:
        pickle_out = open(summaries_path + 'dbn_train_params.pickle', 'wb')
        pickle.dump(dbn_train_params, pickle_out)
        pickle_out.close()


if __name__ == '__main__':
    main()
