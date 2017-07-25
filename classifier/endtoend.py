# -*- coding: utf-8 -*-
'''End-to-end deep learning.
   This allows you to slurp the whole file into GPU memory by breaking it apart into chunks
'''
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation, TimeDistributed, Embedding, AveragePooling1D, GlobalAveragePooling2D, Conv1D, ELU
from keras.regularizers import l2

charset = set(range(257))  # ord(256) = "EOF" (different from ord(0))


def create_model(input_shape, byte_embedding_size=2, input_dropout=0.05, hidden_dropout=0.1, kernel_size=16, n_filters_per_layer=[64, 256, 1024], n_mlp_layers=2):
    '''End to end deep learning.

    Note that to fit in memory of most graphics cards, the input file must be split up into multiple chunks, which will take
    up contiguous memory on the GPU card.  Then we process each chunk separately and combine within the model.  Note
    that this isn't exactly the same as having operated on the whole file, because of boundary artifacts of each chunk.
    But (shrug). It works.

    Args:
        input_shape (tuple) : input shape to the model. For this model, should be of shape (file_chunks, chunk_size)
        byte_embedding_size (int): each byte is embedded into a space of this dimension
        input_dropout (float): fraction of input features to drop out during training        
        kernel_size (int): kernel size for 1-D filter at each layer
        n_filters_per_layer (tuple): number of convolutional filters per layer
        n_mlp_layers (int): number of hidden layers for the final multilayer perceptron

    Returns:
        keras.models.Sequential : a model to train
    '''
    file_chunks, chunk_size = input_shape

    model = Sequential()

    # first, we'll represent bytes in some embedding space.
    # if byte_embedding_size=2, then each byte, like 'A' will be mapped to a point in 2d space
    # the mapping is learned end-to-end to optimize performance

    # TimeDistributed is operating on each chunk
    model.add(TimeDistributed(Embedding(len(charset), byte_embedding_size,
                                        input_length=chunk_size, name="embedding"), input_shape=(file_chunks, chunk_size)))
    # output shape: (nb_batch, file_chunks, chunk_size, byte_embedding_size)

    # dropout the input to prevent overfitting to any one feature
    # (a similar concept to randomization in random forests,
    #   but we choose less severe feature sampling  )
    # drop out the entire symbol
    model.add(Dropout(input_dropout, noise_shape=(file_chunks, chunk_size, 1)))

    # set up hidden layers
    for hidden_layer, n_filters in enumerate(n_filters_per_layer):
        # the layer...activation will come later
        model.add(TimeDistributed(Conv1D(n_filters, kernel_size, name='conv_hidden_{}'.format(
            hidden_layer), kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), padding='valid')))
        # 
        model.add( Dropout(hidden_dropout))
        # batchnormalization helps training
        model.add(BatchNormalization())
        # ...the activation!
        model.add(ELU())

        # summarize and collapse via AveragePooling1D
        # AveragePooling1D instead of MaxPooling1D allows gradients to flow through every branch
        model.add(TimeDistributed(AveragePooling1D(
            pool_size=kernel_size, strides=kernel_size)))

    # output shape: (nb_batch, file_chunks, downsampled_chunk_size, n_filters_per_layer[-1] )

    # get max response of filter within each chunk *and* over all chunks
    model.add(GlobalAveragePooling2D())
    # (nb_batch, n_filters_per_layer[-1])

    # now, time for our fully connected layers (multilayer perceptron)
    for _ in range(n_mlp_layers):
        model.add(
            Dense(n_filters_per_layer[-1], kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(ELU())

    # output layer
    model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))

    # we'll optimize with plain old sgd
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])

    return model
