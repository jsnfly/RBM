import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def treshold_data(data, threshold):
    """
    threshold data with threshold value
    :param data: numpy array
    :param threshold: float
    :return: thresholded data
    """
    data = np.where(data >= threshold, np.ones_like(data), np.zeros_like(data))
    return data


def load_mnist_data():
    """
    Download mnist data if it does not already exist in directory MNIST_Data, else load it from directory.
    """
    try:
        x_train = np.load('MNIST_Data/train_data.npy')
        y_train = np.load('MNIST_Data/train_labels.npy')
        x_test = np.load('MNIST_Data/test_data.npy')
        y_test = np.load('MNIST_Data/test_labels.npy')

        print('MNIST/Data already existed, loaded files.')

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


def main():
    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    # threshold data
    x_train = treshold_data(x_train, 0.5*255)

    # plot one random image to check if the format is correct
    plt.imshow(x_train[np.random.randint(x_train.shape[0]), :].reshape([28, 28]), cmap='binary')
    plt.show()


if __name__ == '__main__':
    main()
