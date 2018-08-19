import evaluate.dimensionless_disrcrimination_value as ddv
from evaluate.helper_functions import *
from evaluate.plot import *
from MNIST_Test.Run_MNIST import load_mnist_data, threshold_data

dbn_path = 'Test1/dbn.pickle'
dbn = load_dbn(dbn_path)
save_path = 'Test1/pca_and_tsne/'

(x_train, y_train), (x_test, y_test) = load_mnist_data()
# threshold data
x_train = threshold_data(x_train, 0.5 * 255)
x_test = threshold_data(x_test, 0.5 * 255)

# make samples indices
np.random.seed(420)
test_indices = make_sample_indices(x_test, 500)

test_samples = x_test[test_indices]
test_labels = y_test[test_indices]

# make pca and tsne of the input layer
pca_samples = calculate_pca(test_samples)
tsne_samples = calculate_tsne(test_samples)

# save plots of pca and tsne of the input layer
pca_plot = plot_pca_or_tsne(pca_samples, test_labels, one_hot_labels=False)
plt.savefig(save_path + 'pca_input_layer.png', format='png')
plt.close()

tsne_plot = plot_pca_or_tsne(tsne_samples, test_labels, one_hot_labels=False)
plt.savefig(save_path + 'tsne_input_layer.png', format='png')
plt.close()

# calculate discrimination value
discrimination_values = []
discrimination_value = ddv.discrimination_value(test_samples,
                                                test_labels,
                                                norm='z')
discrimination_values.append(discrimination_value)

# calculate layerwies activations
layer_activations = layerwise_activations(dbn, test_samples, num_activations=100)

# make and plot pca and tsne for higher layers, calculate discrimination value
for layer_index in range(len(dbn)):
    pca_samples = calculate_pca(layer_activations[layer_index])
    tsne_samples = calculate_tsne(layer_activations[layer_index])

    pca_plot = plot_pca_or_tsne(pca_samples, test_labels, one_hot_labels=False)
    plt.savefig(save_path + 'pca_layer{}.png'.format(layer_index), format='png')
    plt.close()

    tsne_plot = plot_pca_or_tsne(tsne_samples, test_labels, one_hot_labels=False)
    plt.savefig(save_path + 'tsne_layer{}.png'.format(layer_index), format='png')
    plt.close()

    discrimination_value = ddv.discrimination_value(layer_activations[layer_index],
                                                    test_labels,
                                                    norm='z')
    discrimination_values.append(discrimination_value)

plt.plot(discrimination_values)
plt.savefig(save_path + 'discrimination_values.png', format='png')
plt.close()
