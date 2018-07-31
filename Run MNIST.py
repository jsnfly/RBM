import tensorflow as tf
# import numpy as np


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    print('shape training examples: ', x_train.shape)
    print('shape training labels: ', y_train.shape)

    print('shape test examples: ', x_test.shape)
    print('shape test labels: ', y_test.shape)


if __name__ == '__main__':
    main()
