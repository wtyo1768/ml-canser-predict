from cv2 import cv2 as cv2
import tensorflow as tf

filename = './data/Train.tfrecords'
filename_queue = tf.train.string_input_producer([filename], shuffle=True,
                                                num_epochs=1)
reader = tf.TFRecordReader()
key, serialized_example = reader.read(filename_queue)
feature = {
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
    'name': tf.FixedLenFeature([], tf.string)}
img_features = tf.parse_single_example(serialized_example, features=feature)
image = tf.decode_raw(img_features['image_raw'], tf.uint8)
image = tf.reshape(image, [96, 96])
label = tf.cast(img_features['label'], tf.int64)
name = tf.cast(img_features['name'], tf.string)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    val_split = 0
    try:
        print('Transform to jpg ...')
        val_prefix = './feature/validation/'
        train_prefix = './feature/train/'
        while True:
            image_data, label_data, name_data = sess.run([image, label, name])
            if val_split < 4:
                filename = train_prefix + str(label_data) \
                + '/' + bytes.decode(name_data) + '.jpg'
            else:
                filename = val_prefix + str(label_data) \
                + '/' + bytes.decode(name_data) + '.jpg'
            cv2.imwrite(filename, image_data)
            val_split = (val_split + 1) % 6
    except tf.errors.OutOfRangeError:
        print('Done...')
    finally:
        coord.request_stop()
        coord.join(threads)
