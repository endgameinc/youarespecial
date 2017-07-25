import numpy as np
import common
import os
labels = common.fetch_samples()

from sklearn.model_selection import train_test_split
np.random.seed(123)
y_train, y_test, sha256_train, sha256_test = train_test_split(
    list(labels.values()), list(labels.keys()), test_size=1000)

##################
# end-to-end model
# maximum number of bytes we allow per file = 512KB
max_file_length = int(2**19)
file_chunks = 8  # break file into this many chunks
file_chunk_size = max_file_length // file_chunks
batch_size = 8

import endtoend
import math
from keras.callbacks import LearningRateScheduler

# create_model(input_shape, byte_embedding_size=2, input_dropout=0.2, kernel_size=16, n_filters_per_layer=[64,256,1024], n_mlp_layers=2 )
model_e2e = endtoend.create_model(input_shape=(file_chunks, file_chunk_size))
train_generator = common.generator(list(zip(sha256_train, y_train)), batch_size, file_chunks, file_chunk_size)
test_generator = common.generator(list(zip(sha256_test, y_test)), 1, file_chunks, file_chunk_size)
model_e2e.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(len(sha256_train) / batch_size),
                        epochs=20,
                        callbacks=[LearningRateScheduler(
                            lambda epoch: common.schedule(epoch, start=0.1, decay=0.5, every=1)
                            )
                        ],
                        validation_data=test_generator,
                        validation_steps=len(sha256_test))
y_pred = []
for sha256, lab in zip(sha256_test, y_test):
    y_pred.append(
        model_e2e.predict_on_batch(
            np.asarray([get_file_data(sha256, lab)]).reshape(
                (-1, file_chunks, file_chunk_size))
        )
    )
common.summarize_performance(np.asarray(
    y_pred).flatten(), y_test, "End-to-end convnet")
model_e2e.save('endtoend.h5')
