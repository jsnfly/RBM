# SHAPE IN MAKE DS

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import train.make_datasets as make_ds

from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from evaluate.helper_functions import *


# Load train files:
load_path = '/home/jonas/HDD/data/unwindowed/z-transformed_shuffled/'

keys = ['sample', 'one_hot_label']
data_types = ['float32', 'int32']

training_files = []
validation_files = []
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

validation_files = sorted(validation_files)
training_files = sorted(training_files)

print(len(training_files))
print(len(validation_files))

batch_size = 128
train_dataset = make_ds.dataset_from_TFRecords(training_files, batch_size,
                                               keys,
                                               data_types,
                                               shuffle_buffer=40000,
                                               parallel_reads=len(training_files),
                                               num_cores=8)
train_dataset = train_dataset.repeat()

test_dataset = make_ds.dataset_from_TFRecords(validation_files, batch_size,
                                              keys,
                                              data_types,
                                              shuffle_buffer=40000,
                                              parallel_reads=len(validation_files),
                                              num_cores=8)
test_dataset = test_dataset.repeat()

# set up model
model = Sequential()
model.add(Conv1D(64, 16, activation='relu', input_shape=(1*256, 3, )))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Conv1D(64, 16, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Conv1D(32, 8, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.summary()

num_classes = 5
optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train
epochs = 50
save_path = '/home/jonas/PycharmProjects/convnet/test4/'
callbacks = [ModelCheckpoint(save_path + 'Model.hdf5', save_best_only=True, verbose=1)]

history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=8000,
                    validation_data=test_dataset,
                    validation_steps=1200, verbose=2, callbacks=callbacks)

pickle_out = open(save_path + 'History' + '.pickle', 'wb')
pickle.dump(history.history, pickle_out)
pickle_out.close()

K.clear_session()
