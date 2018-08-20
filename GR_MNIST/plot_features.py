import numpy as np
import matplotlib.pyplot as plt
import os
from evaluate.helper_functions import load_dbn, downward_propagate_features
from pathlib import Path


def main():
    dbn_path = Path('Test_finetune_smaller_layers_no_scale/')
    dbn_path = os.fspath(dbn_path / 'dbn.pickle')
    dbn = load_dbn(dbn_path)

    save_path = 'Test_finetune_smaller_layers_no_scale/features/'
    for layer_index in range(len(dbn)):
        layer = dbn[layer_index]
        inds = np.random.choice(range(layer.num_hunits), size=16, replace=False)

        # plot features with downward propagate approach
        receptive_fields = downward_propagate_features(dbn, layer_index, inds,
                                                       num_runs=1,
                                                       num_neurons=4,
                                                       activation_value=100.0)
        fig, axes = plt.subplots(4, 4, figsize=(2.895, 2.895))
        for c, ax in enumerate(axes.flat):
            receptive_field = receptive_fields[c]
            ax.imshow(receptive_field.reshape([28, 28]), cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        if save_path is not None:
            plt.savefig(save_path + 'receptive_fields_layer{}.png'.format(layer_index), format='png')
        plt.show()

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


if __name__ == '__main__':
    main()
