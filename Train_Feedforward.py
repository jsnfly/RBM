from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def feedforward_network(layer_sizes, dropout_rates, activation='sigmoid'):
    model = Sequential()
    for li, ls in enumerate(layer_sizes):
        model.add(Dense(units=ls, activation = activation))
        if dropout_rates[li] != 0:
            model.add(Dropout(rate = dropout_rates[li]))
    return model


layer