# coding:utf-8
# 模型数据输入模块

import tensorflow as tf
import cifar10_model
from cifar10_config import *
# import cifar10_model_test


with tf.Graph().as_default():
    files = tf.train.match_filenames_once("./cifar10_data/data_batch_*")
    image_batch, label_batch = cifar10_model.cifar10_input(files)
    logits = cifar10_model.inference(image_batch)
    loss = cifar10_model.loss(logits, label_batch)
    global_step = tf.Variable(0, trainable=False)
    decay_learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                            100, 0.96, staircase=True,
                                                     name='decay_learning_rate')
    with tf.name_scope('decay_learning_rate'):
        tf.summary.scalar("learning_rate", decay_learning_rate)
    train_op = tf.train.AdamOptimizer(decay_learning_rate).minimize(loss,global_step=global_step)

    # test_image, test_label = cifar10_model_test.cifar10_test_data()
    # test_label = tf.reshape(test_label, [batch_size_test])
    # label_predicted = cifar10_model.inference(test_image)
    label_batch = tf.cast(tf.reshape(label_batch, [batch_size]), tf.int64)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), label_batch)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    # 神经网络搭建
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()  # 局部变量初始化，因为tf.match_filename_one需要
        sess.run(files)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(train_step):
            summary,_ = sess.run([merged,train_op])
            summary_writer.add_summary(summary, i)
        summary_writer.close()
        coord.request_stop()
        coord.join(threads)
