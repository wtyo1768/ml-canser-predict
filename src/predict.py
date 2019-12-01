from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd


IMG_HEIGHT = 64
IMG_WIDTH = 64
layers = tf.keras.layers

model = tf.keras.models.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu',
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("loading weights...")
model.load_weights('./model.h5')
print("start predict...")

TestGen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_dir = './data/test_feature'
TestData=TestGen.flow_from_directory(directory=test_dir,
                                     shuffle=False,
                                     batch_size=1,
                                     class_mode=None,
                                     target_size=(IMG_HEIGHT, IMG_WIDTH))
TestData.reset()
predict = model.predict_generator(TestData)
predict = predict > 0.5
predict = predict.flatten()
#for i in range(len(predict)):
 #   if predict[i]==True:
  #      predict[i] = int(1)
  #  else:
   #     predict[i] = int(0)a
predict = predict.astype(int)
fname = TestData.filenames
fname = [item.split('/')[1].split('.')[0] for item in fname]
df = pd.DataFrame({
    "id":fname,
    "label":predict
})
df.to_csv('./predict.csv', index=False)

