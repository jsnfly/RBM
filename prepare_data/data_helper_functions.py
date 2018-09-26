import tensorflow as tf
import numpy as np
import math


def txtfile_to_nparray(file_path, start):
    """
    helper function to extract data from txtfiles
    :param file_path:
    :param start: number of lines to skip in the beginning
    :return: np.array with data from file
    """
    file = open(file_path, 'r', encoding="ISO-8859-1").read().split('\n')[int(start):]
    values = []
    for line in file:
        if len(line) > 0:
            values.append(int(line))
    array = np.asarray(values)
    return array


def h(n, N):
    """
    hamming window function to reduce spectral leakage
    n: sample point
    N: total number of sample points in interval
    """
    return 0.54 - 0.46 * tf.cos(2 * math.pi * n / (tf.cast(N, tf.float32) - 1))


def fourier_trafo(sample):
    """
    fourier transformation of sample
    """
    fast_fourier_transform = tf.fft(tf.cast(sample, tf.complex64))
    magnitudes = tf.sqrt(tf.square(tf.real(fast_fourier_transform)) + tf.square(tf.imag(fast_fourier_transform)))
    fourier = 2 * tf.split(magnitudes, num_or_size_splits=2, axis=1)[0] / tf.cast(tf.shape(sample)[0], tf.float32)
    return fourier


def make_fourier_trafo_sample(sample, label):
    """
    make fourier trafo for a sample
    Args:
        sample: sample, needs to be in format (ch1_samples,ch2_samples,ch3_samples)
        label: corresponding label
    """
    window_size = tf.cast(tf.shape(sample)[1] / 3, tf.int32)
    sample_ch1 = sample[:, :window_size] * h(tf.cast(tf.range(0, window_size), tf.float32), window_size)
    sample_ch2 = sample[:, window_size:2 * window_size] * h(tf.cast(tf.range(0, window_size), tf.float32), window_size)
    sample_ch3 = sample[:, 2 * window_size:] * h(tf.cast(tf.range(0, window_size), tf.float32), window_size)

    #####################################
    fourier_ch1 = fourier_trafo(sample_ch1)[:, :40]
    fourier_ch2 = fourier_trafo(sample_ch2)[:, :40]
    fourier_ch3 = fourier_trafo(sample_ch3)[:, :40]
    #####################################

    fourier_sample = tf.concat([fourier_ch1, fourier_ch2, fourier_ch3], axis=1)

    return fourier_sample, label
