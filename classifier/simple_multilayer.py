# -*- coding: utf-8 -*-
'''A simple multilayer perceptron
'''
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation, ELU


def create_model(input_shape, hidden_layers=[1024, 512, 256], input_dropout=0.1, hidden_dropout=0.5):
    '''Define a simple multilayer perceptron.

    Args:
        input_shape (tuple): input shape to the model. For this model, should be of shape (dim,)
        input_dropout (float): fraction of input features to drop out during training
        hidden_layers (tuple): a tuple/list with number of hidden units in each hidden layer

    Returns:
        keras.models.Sequential : a model to train
    '''
    model = Sequential()

    # dropout the input to prevent overfitting to any one feature
    # (a similar concept to randomization in random forests,
    #   but we choose less severe feature sampling  )
    model.add(Dropout(input_dropout, input_shape=input_shape))

    # set up hidden layers
    for n_hidden_units in hidden_layers:
        # the layer...activation will come later
        model.add(Dense(n_hidden_units))
        # dropout to prevent overfitting
        model.add(Dropout(hidden_dropout))
        # batchnormalization helps training
        model.add(BatchNormalization())
        # ...the activation!
        model.add(ELU())

    # the output layer
    model.add(Dense(1, activation='sigmoid'))

    # we'll optimize with plain old sgd
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])

    return model
