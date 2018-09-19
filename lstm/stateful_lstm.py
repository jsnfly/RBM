import sys
sys.path.append('..')
import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from lstm_helper_functions import *
from tensorflow.python.client import device_lib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(device_lib.list_local_devices()

# get training and validation files
load_path = '/home/jonas/HDD/data/unwindowed/unwindowed_Fourier-transformed/'
keys = ['sample', 'one_hot_label']
data_types = ['float32', 'int32']

training_files = []
validation_files = []

# only works if there is now dataset number higher than 50!
for file in os.listdir(load_path):
    if 'dataset5' in file:
        validation_files.append(load_path + file)
    elif 'dataset10' in file:
        validation_files.append(load_path + file)
    elif 'dataset15' in file:
        validation_files.append(load_path + file)
    elif 'dataset20' in file:
        validation_files.append(load_path + file)
    elif 'dataset25' in file:
        validation_files.append(load_path + file)
    else:
        training_files.append(load_path + file)

training_files = sorted(training_files)
validation_files = sorted(validation_files)
print(len(training_files))
print(len(validation_files))

# extract samples from files
all_train_samples, all_train_labels, all_val_samples, all_val_labels = extract_samples(training_files,
                                                                                       validation_files,
                                                                                       keys,
                                                                                       data_types)

print('after file reading:\n___________________')
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print('val files: ')
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

num_time_steps = 60
LSTM_size = 64
num_x_signals = 120
num_labels = 5
batch_size = 64

# cut all datasets to have a (whole number * batch_size * time_steps) samples:
num_samples_batch = batch_size * num_time_steps

all_train_samples = [train_samples[:train_samples.shape[0] // num_samples_batch * num_samples_batch, :] for
                     train_samples in all_train_samples]

all_train_labels = [train_labels[:train_labels.shape[0] // num_samples_batch * num_samples_batch, :] for
                    train_labels in all_train_labels]

all_val_samples = [val_samples[:val_samples.shape[0] // num_samples_batch * num_samples_batch, :] for
                   val_samples in all_val_samples]

all_val_labels = [val_labels[:val_labels.shape[0] // num_samples_batch * num_samples_batch, :] for
                  val_labels in all_val_labels]

print('after cutting of unusable samples:\n___________________')
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print('val files: ')
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

# reshape to match shape (num_series,time_steps,num_x_signals):
all_train_samples = [np.reshape(train_samples, [train_samples.shape[0] // num_time_steps,
                                                num_time_steps,
                                                num_x_signals]) for train_samples in all_train_samples]

all_train_labels = [np.reshape(train_labels, [train_labels.shape[0] // num_time_steps,
                                              num_time_steps,
                                              num_labels]) for train_labels in all_train_labels]

all_val_samples = [np.reshape(val_samples, [val_samples.shape[0] // num_time_steps,
                                            num_time_steps,
                                            num_x_signals]) for val_samples in all_val_samples]

all_val_labels = [np.reshape(val_labels, [val_labels.shape[0] // num_time_steps,
                                          num_time_steps,
                                          num_labels]) for val_labels in all_val_labels]

print('after reshaping samples:\n___________________')
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print('val files: ')
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

for counter, train_samples in enumerate(all_train_samples):
    batches = make_batches(train_samples, batch_size)
    train_samples = np.concatenate(batches)
    all_train_samples[counter] = train_samples

for counter, train_labels in enumerate(all_train_labels):
    batches = make_batches(train_labels, batch_size)
    train_labels = np.concatenate(batches)
    all_train_labels[counter] = train_labels

for counter, val_samples in enumerate(all_val_samples):
    batches = make_batches(val_samples, batch_size)
    val_samples = np.concatenate(batches)
    all_val_samples[counter] = val_samples

for counter, val_labels in enumerate(all_val_labels):
    batches = make_batches(val_labels, batch_size)
    val_labels = np.concatenate(batches)
    all_val_labels[counter] = val_labels

print('after making batches :\n___________________')
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print('val files: ')
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

lengths_train_datasets = [d.shape[0] for d in all_train_samples]
lengths_val_datasets = [d.shape[0] for d in all_val_samples]

epochs = 500

train_accuracies = []
test_accuracies = []
current_best_val_loss = 100

model = Sequential()
model.add(LSTM(units=LSTM_size, return_sequences=True, batch_input_shape=(batch_size, num_time_steps, num_x_signals),
               dropout=0.50, recurrent_dropout=0, stateful=True))
model.add(TimeDistributed(Dense(5, activation='softmax')))

optimizer = Adam(0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

save_path = '/home/jonas/PycharmProjects/LSTM_trails/stateful/'
save_path = os.path.join(save_path, 'Fourier_{}timesteps_{}ls_{}bs'.format(num_time_steps, LSTM_size, batch_size))
if not os.path.exists(save_path):
    os.makedirs(save_path)

for epoch in range(epochs):
    tr_losses = []
    tr_accs = []
    for d, length in enumerate(lengths_train_datasets):
        train_samples = all_train_samples[d]
        train_labels = all_train_labels[d]
        num_batches = length // batch_size
        for i in range(num_batches):
            x_batch = train_samples[i * batch_size:(i + 1) * batch_size]
            y_batch = train_labels[i * batch_size:(i + 1) * batch_size]
            tr_loss, tr_acc = model.train_on_batch(x_batch, y_batch)
            tr_losses.append(tr_loss)
            tr_accs.append(tr_acc)
        model.reset_states()
    train_accuracies.append(np.mean(tr_accs))

    test_losses = []
    test_accs = []
    for d, length in enumerate(lengths_val_datasets):
        val_samples = all_val_samples[d]
        val_labels = all_val_labels[d]
        num_samples = length // batch_size
        for i in range(num_samples):
            x_batch = val_samples[i * batch_size:(i + 1) * batch_size]
            y_batch = val_labels[i * batch_size:(i + 1) * batch_size]
            test_loss, test_acc = model.test_on_batch(x_batch, y_batch)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        model.reset_states()
    test_accuracies.append(np.mean(test_accs))

    if np.mean(test_losses) < current_best_val_loss:
        current_best_val_loss = np.mean(test_losses)
        model.save(os.path.join(save_path, 'best_model'))
        print('Model saved! Val acc: ', np.mean(test_accs))

    else:
        print('Model not saved, Val acc: ', np.mean(test_accs))

pickle_out = open(os.path.join(save_path, 'train_accuracies'), 'wb')
pickle.dump(train_accuracies, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(save_path, 'test_accuracies'), 'wb')
pickle.dump(test_accuracies, pickle_out)
pickle_out.close()

K.clear_session()
