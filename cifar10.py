# coding:utf-8
# 模型数据输入模块

import tensorflow as tf
import cifar10_model
from cifar10_config import *
import cifar10_model_test


with tf.Graph().as_default():
    files = tf.train.match_filenames_once("./cifar10_data/data_batch_*")
    image_batch, label_batch = cifar10_model.cifar10_input(files)
    logits = cifar10_model.inference(image_batch)
    loss = cifar10_model.loss(logits, label_batch)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    test_image, test_label = cifar10_model_test.cifar10_test_data()
    test_label = tf.reshape(test_label, [batch_size_test])
    label_predicted = cifar10_model_test.inference_test(test_image)
    test_label = tf.cast(tf.reshape(label_batch, [batch_size]), tf.int64)
    correct_prediction = tf.equal(tf.argmax(logits, 1), test_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 神经网络搭建
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()  # 局部变量初始化，因为tf.match_filename_one需要
        sess.run(files)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(train_step):
            print("train step is ", i)
            if i % 50 == 0:
                print(sess.run(accuracy))
            sess.run(train_op)

        coord.request_stop()
        coord.join(threads)
