import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusion_matrix)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
