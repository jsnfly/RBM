import pickle
import numpy as np
import matplotlib.pyplot as plt
from plot_features import load_dbn
import dimensionless_disrcrimination_value as ddv

load_path = '/home/jonas/HDD/DBN_MNIST_runs/Runs_equal_size_NEW/run_80/'

dbn = load_dbn(load_path + 'DBN.pickle')

pickle_in = open(load_path + 'activations.pickle', 'rb')
activations = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('/home/jonas/HDD/DBN_MNIST_runs/Runs_equal_size_NEW/' + 'PCA_indices.pickle', 'rb')
PCA_indices = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('/home/jonas/PycharmProjects/RBM/old_MNIST_Data/'+'test_labels.pickle', 'rb')
test_labels = pickle.load(pickle_in).astype(np.float32)
pickle_in.close()

test_classes = np.argmax(test_labels[PCA_indices, :], axis=1)

li = 5
fig, axes = plt.subplots(5, 2)
fig.subplots_adjust(hspace=0.0, wspace=0.0)
for c, ax in enumerate(axes.flat):
    class_indices = np.where(test_classes == c)[0]
    class_activations = activations['layer_{}'.format(li)][class_indices]
    print(class_activations.shape)
    ax.imshow(np.mean(class_activations, axis=0).reshape(10, 50))
    # ax.set_title('layer_{}, class_{}'.format(li, c))
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.colorbar()
plt.tight_layout()
plt.show()

discrimination_v = ddv.discrimination_value(activations['layer_{}'.format(li)], test_classes, 'z')
print(discrimination_v)