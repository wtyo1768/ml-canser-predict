from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from cv2 import cv2 as cv2
import os

def get_File(file_dir):
    image_list = []
    label_list = []
    df = pd.read_csv('./data/label/train_labels.csv')
    for idx, row in df.iterrows():
        img_path = file_dir + row['id'] +'.tif'
        image_list.append(img_path)
        label_list.append(row['label'])
    return image_list, label_list
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_TFRecord(images, filename, labels ):
    n_samples = len(images)
    TFWriter = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start...')
    for i in np.arange(0, n_samples):
        name = images[i].split('/')[3].split('.')[0]
        try:
            image = cv2.imread(images[i], 0)
            image_raw = image.tostring()
            name = str.encode(name)
            ftrs = tf.train.Features(
                    feature={
                        'image_raw': bytes_feature(image_raw),
                        'label' : int64_feature(labels[i]),
                        'name' : bytes_feature(name),
                    })
            example = tf.train.Example(features=ftrs)
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n',e)
    TFWriter.close()
    print('Transform done!')

import pandas as pd

if '__main__' == __name__:
    image_list , label_list = get_File('./data/train_data/')
    convert_to_TFRecord(image_list , './data/Train.tfrecords' , label_list)
