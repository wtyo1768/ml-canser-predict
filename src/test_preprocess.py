from __future__ import absolute_import, division,\
                        print_function, unicode_literals
import tensorflow as tf
import numpy as np
from cv2 import cv2 as cv2
import pandas as pd


def get_File(file_dir):
    image_list = []
    df = pd.read_csv('./data/label/test1_label.csv')
    for idx, row in df.iterrows():
        img_path = file_dir + row['id'] + '.tif'
        image_list.append(img_path)
    return image_list


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_TFRecord(images, filename):
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
                        'name': bytes_feature(name),
                    })
            example = tf.train.Example(features=ftrs)
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n', e)
    TFWriter.close()
    print('Transform done!')


def convert_to_jpg():
    filename = './data/Test.tfrecords'
    filename_queue = tf.train.string_input_producer([filename], shuffle=True,
                                                num_epochs=1)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    feature = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'name': tf.FixedLenFeature([], tf.string)}
    img_features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [96, 96])
    name = tf.cast(img_features['name'], tf.string)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        count = 0
        try:
            print('Start transform to jpg...')
            while True:
                image_data, name_data = sess.run([image, name])
                filename = './data/test_feature/' + bytes.decode(name_data)  + '.jpg'
                cv2.imwrite(filename, image_data)
        except tf.errors.OutOfRangeError:
            print('Done...')
        finally:
            coord.request_stop()
            coord.join(threads)

if '__main__' == __name__:
    #image_list  = get_File('./data/test1_data/')
    #convert_to_TFRecord(image_list, './data/Test.tfrecords')
    convert_to_jpg()
