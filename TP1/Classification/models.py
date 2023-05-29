import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten

# Define a function to create and train the model
def create_softmax_model(X_train, y_sparse_train, X_test, y_sparse_test, learning_rate, batch_size, optimizer, dropout_rate, batch_normalization, weight_initialization):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    if batch_normalization:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax', kernel_initializer=weight_initialization))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer(learning_rate=learning_rate), metrics=['accuracy'])
    history = model.fit(X_train, y_sparse_train, batch_size=batch_size, epochs=10, verbose=0, validation_data=(X_test, y_sparse_test))
    return history

def create_mlp_model(X_train, y_sparse_train, X_test, y_sparse_test, learning_rate, batch_size, optimizer, activation, dropout_rate, batch_normalization, weight_initialization):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(Dense(256, activation=activation, kernel_initializer=weight_initialization))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer=weight_initialization))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer(learning_rate=learning_rate), metrics=['accuracy'])
    history = model.fit(X_train, y_sparse_train, batch_size=batch_size, epochs=10, verbose=0, validation_data=(X_test, y_sparse_test))
    return history

# Define a function to plot the accuracy against a hyperparameter
def plot_accuracy(hyperparameter_values, accuracies, hyperparameter_name, log_scale=False):
    plt.plot(hyperparameter_values, accuracies, marker='o')
    plt.title(f'Accuracy vs {hyperparameter_name}')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('Accuracy')
    if log_scale:
        plt.xscale('log')
    plt.grid(True)
    plt.show()

# Define a function to plot the loss against a hyperparameter
def plot_loss(hyperparameter_values, losses, hyperparameter_name, log_scale=False):
    plt.plot(hyperparameter_values, losses, marker='o')
    plt.title(f'Loss vs {hyperparameter_name}')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('Loss')
    if log_scale:
        plt.xscale('log')
    plt.grid(True)
    plt.show()