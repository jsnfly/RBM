from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def feedforward_network(layer_sizes, dropout_rates, num_classes, optimizer, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(units=layer_sizes[1], activation=activation, input_shape=(layer_sizes[0],)))
    model.add(Dropout(rate=dropout_rates[0]))
    for li, ls in enumerate(layer_sizes[:2]):
        li += 2
        model.add(Dense(units=ls, activation=activation))
        if dropout_rates[li-1] != 0:
            model.add(Dropout(rate=dropout_rates[li-1]))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
