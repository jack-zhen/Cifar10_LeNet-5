# coding: utf-8

import tensorflow as tf
from cifar10_config import *


def cifar10_test_data():
    # 构造输入数据流
    test_file =[b"./cifar10_data/test_batch.bin"]
    test_filename_queue = tf.train.string_input_producer(test_file, shuffle=False)
    test_reader = tf.FixedLengthRecordReader(record_bytes=3073)  # 3073:label(1)+ width(32)*high(32)*channel(3)
    _, value = test_reader.read(test_filename_queue)
    test_record_bytes = tf.decode_raw(value, tf.uint8)
    test_image_label_temp = tf.slice(test_record_bytes, [0], [1])
    test_image_label = tf.cast(test_image_label_temp, tf.int32)
    test_image_extracted_temp = tf.slice(test_record_bytes, [1], [3072])
    test_image_extracted = tf.reshape(test_image_extracted_temp, [3, 32, 32])
    image_int8 = tf.transpose(test_image_extracted, [1, 2, 0])
    test_image_value = tf.cast(image_int8, tf.float32)
    test_image_value.set_shape([32, 32, 3])
    test_image_label.set_shape([1])
    # 构造组训练集
    test_image, test_label = tf.train.batch([test_image_value, test_image_label],
                                              batch_size=batch_size_test, num_threads=2,
                                              capacity=100+batch_size_test*3)
    return test_image, test_label


def biases_generator(name, shape, initializer):
    biases = tf.get_variable(name, shape=shape, initializer=initializer)
    return biases


def weights_generator(name, shape, initializer):
    weights = tf.get_variable(name, shape=shape, initializer=initializer, )
    return weights


def inference_test(images):
    # 第一层巻积
    with tf.variable_scope('conv1') as scope:
        scope.reuse_variables()
        kernel = tf.get_variable('weights', shape=[5, 5, 3, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
                                 dtype=tf.float32)
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.))
        conv1_temp = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1_temp, biases), name=scope.name)
        scope.reuse_variables()

        # print("After  conv1 data shape",conv1.shape)
    # 池化层1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # print("After  pool1 data shape", pool1.shape)

    # 第二层巻积
    with tf.variable_scope('conv2') as scope:
        scope.reuse_variables()
        kernel = tf.get_variable('weights', shape=[5, 5, 64, 128],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
                                 dtype=tf.float32)
        conv2_temp = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.relu(tf.nn.bias_add(conv2_temp, biases))

        # print("After  conv2 data shape", conv2.shape)
    # 池化层2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # print("After  pool2 data shape", pool2.shape)

    # 全链接层1
    with tf.variable_scope('local3') as scope:
        scope.reuse_variables()
        reshape = tf.reshape(pool2, [batch_size_test, -1])
        dim = reshape.shape[1].value
        weights = weights_generator('weights3', [dim, 384],
                                    tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
        biases = biases_generator('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        # print("After local3 data shape", local3.shape)

    # 全链接层2
    with tf.variable_scope('local4') as scope:
        scope.reuse_variables()
        weights = weights_generator('weights4', [384, 192],
                                    tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
        biases = biases_generator('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        # print("After local4 data shape", local4.shape)

    # Softmax 分类层
    with tf.variable_scope('softmax_linear') as scope:
        scope.reuse_variables()
        weights = weights_generator('weights', [192, 10],
                                    tf.truncated_normal_initializer(stddev=1 / 192.0, dtype=tf.float32))
        biases = biases_generator('biases', [10], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        # print("After softmax_linear data shape", softmax_linear.shape)
        return softmax_linear
