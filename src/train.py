from __future__ import absolute_import, division,\
                        print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

batch_size = 128
epochs = 15
IMG_HEIGHT = 64
IMG_WIDTH = 64

train_dir = './feature/train'
test_dir = './feature/validation'

train_img_gen =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                 horizontal_flip=True,
                                                                 rotation_range=45,
                                                                 zoom_range=0.5)
train_data_gen = train_img_gen.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

test_img_gen =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = train_img_gen.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same',
                           activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensor_board = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

#model.summary()
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(train_data_gen.filenames) // epochs,
    validation_data=test_data_gen,
    epochs=epochs,
    validation_steps=len(test_data_gen.filenames) // epochs,
    callbacks=[tensor_board],
    verbose=1
)
model.save('./model/model.h5')

