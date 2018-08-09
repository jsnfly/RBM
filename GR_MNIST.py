import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from RBM import RBM
import os
from pathlib import Path
from Run_MNIST import load_mnist_data
from plot_features import load_dbn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # load data
    (x_train, y_train), (_, _) = load_mnist_data()

    # plot one random image to check if the format is correct
    plt.imshow(x_train[np.random.randint(x_train.shape[0]), :].reshape([28, 28]), cmap='binary')
    plt.show()

    # Set up Deep Belief Network
    layer_sizes = [784, 512, 64, 64]
    layer_types = ['gr', 'gr', 'gr']

    mnist_dbn = generate_dbn(layer_sizes, layer_types)

    # # for loading old dbn for further training:
    # # load dbn
    # mnist_dbn = load_dbn('GR_MNIST1\\dbn.pickle')
    # # reinitialize layers to retrain
    # mnist_dbn['layer_2'] = RBM(64, 64, 'gr', 2)

    # Training parameters
    dbn_train_params = {
        'epochs': [20, 20, 20],  # number of training epochs
        'batch_size': [128, 128, 128],  # size of one training batch
        'cd_steps': [1, 1, 1],  # number of CD training steps
        'update_vbiases': [True, True, True],  # if false vbiases are set to zero throughout the training
        'learning_rate': [0.005, 0.005, 0.005],  # learning rate at the begin of training
        'lr_decay': [(4, 0.5), (4, 0.5), (4, 0.5)],  # decay of learning rate (every epochs,decay factor)
        'summary_frequency': [250, 250, 250],  # write to summary every x batches
        'sparsity_rate': [0.05, 0.04, 0.03],  # rate with which sparsity is enforced
        'sparsity_goal': [1.0, 1.0, 1.0]  # goal activation probability
    }

    # Train DBN
    summaries_path = Path('GR_MNIST_sparse1_2/')
    for li in range(0, len(mnist_dbn)):
        # set up layer summary path
        summary_path = summaries_path / 'layer_{}'.format(li)

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
                        summary_path=os.fspath(summary_path),
                        summary_frequency=summary_frequency,
                        dbn=mnist_dbn,
                        update_vbiases=update_vbiases,
                        start_learning_rate=start_learning_rate,
                        learning_rate_decay=lr_decay,
                        cd_steps=cd_steps,
                        sparsity_rate=sparsity_rate,
                        sparsity_goal=sparsity_goal)

    # save parameter dicts:
    pickle_out = open(summaries_path / 'dbn_train_params.pickle', 'wb')
    pickle.dump(dbn_train_params, pickle_out)
    pickle_out.close()

    # save trained dbn:
    pickle_out = open(summaries_path / 'dbn.pickle', 'wb')
    pickle.dump(mnist_dbn, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    main()
