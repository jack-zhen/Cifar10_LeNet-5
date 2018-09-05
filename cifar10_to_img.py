#coding:utf-8
import tensorflow as tf
import scipy.misc
import math

#文件名队列
filename = ['cifar10_data/data_batch_1.bin','cifar10_data/data_batch_2.bin','cifar10_data/data_batch_3.bin']
filename_quene = tf.train.string_input_producer(filename)
#reader
reader = tf.FixedLengthRecordReader(record_bytes=3073)
key,value = reader.read(filename_quene)
record_bytes = tf.decode_raw(value, tf.uint8) #读取出来的是string，转换为int8
image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
image_extracted = tf.reshape(tf.slice(record_bytes, [1], [3072]),[3,32,32])
image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
reshaped_image = tf.cast(image_uint8image, tf.float32)
    

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners(sess=sess)
    
    #保存30张图片
    for i in range(30):

        #quene运行慢汇报错，防止报错让CPU忙一会
        for j in range(1000000):
            a=math.sqrt(j)
        image_arry = sess.run(reshaped_image)
        scipy.misc.toimage(image_arry).save('img/test%d.jpg'%i)
    
