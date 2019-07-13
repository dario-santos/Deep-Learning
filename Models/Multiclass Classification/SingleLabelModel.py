from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# 1 - Loads the data
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.reuters.load_data(num_words=10000)


# 2 - Processing of the raw data to attend the requisites of our network
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = tf.keras.utils.to_categorical(train_labels)
y_test = tf.keras.utils.to_categorical(test_labels)


# 3 - The construction of our model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10000, )),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(46, activation='softmax')
])


# 4 - Otimizador, loss function and metrics
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 5 - Train and validation
history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))

