# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:27:10 2018

@author: cmy

"""

import tensorflow as tf 
import res_block as res

from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_train"
logdir = "{}/run-{}/".format(root_logdir, now)
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Graph().as_default() as g:
    with tf.name_scope('input'):
        x_input=tf.placeholder(tf.float32,[None,784],name='input')
        y_label=tf.placeholder(tf.float32,[None,10],name='label')
        keep_prob = tf.placeholder(tf.float32,name='prob')
        x_image = tf.reshape(x_input, [-1, 28, 28, 1],name='image')
        
    with tf.name_scope('conv1'):
        W_conv1 = res.weight_variable([3,3, 1,64],name='layer1') # patch 3x3, in size 1, out size 32
        b_conv1 = res.bias_variable([64],name='layer1')
        h_conv1 = tf.nn.relu(res.conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
        result=h_conv1
        
    
    res_result1=res.res_block(x_input=result,kernel_size=3,in_filter=64,out_filter=[64,64,64],stage=1,training=False)
    
    with tf.name_scope('pool1'):
        h_pool1=res.max_pool_2x2(res_result1)
    
    res_result2=res.res_block(x_input=h_pool1,kernel_size=3,in_filter=64,out_filter=[64,64,64],stage=2,training=False)
    
    with tf.name_scope('conv2'):
        W_conv2 = res.weight_variable([3,3,64,32],name='layer1') # patch 3x3, in size 1, out size 32
        b_conv2 = res.bias_variable([32],name='layer1')
        h_conv2 = tf.nn.relu(res.conv2d(res_result2, W_conv2) + b_conv2) # output size 28x28x32
    with tf.name_scope('pool2'):
        h_pool2 = res.max_pool_2x2(h_conv2)    
        #result3=h_pool2
    
    with tf.name_scope('fc1'):
        W_fc1 = res.weight_variable([7*7*32, 1024],name='fc1')
        b_fc1 = res.bias_variable([1024],name='fc1')
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        #h_fc1_drop=h_fc1
   
    with tf.name_scope('fc2'):
        ## func2 layer ##
        W_fc2 = res.weight_variable([1024, 10],name='fc2')
        b_fc2 = res.bias_variable([10],name='fc2')
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    with tf.name_scope('loss'):
    # the error between prediction and real data
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
        tf.summary.scalar('loss_function',cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
    with tf.name_scope('eval'):
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy_function',accuracy)
    
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    
with tf.Session(graph=g) as sess:       
    # important step
    sess.run(init)
    file_writer = tf.summary.FileWriter(logdir, sess.graph) 
    for i in range(5000+1):        
        X_batch=mnist.train.images
        Y_batch=mnist.train.labels
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_loss=sess.run(train_step, feed_dict={x_input: batch_xs, y_label: batch_ys, keep_prob:0.5})
       
        if i % 50 == 0:
             print('training loss:',sess.run(cross_entropy,feed_dict={x_input: batch_xs, y_label: batch_ys,keep_prob:0.5}))
             train_acc=sess.run(accuracy,feed_dict={x_input:batch_xs,y_label:batch_ys,keep_prob:1})
             test_acc=sess.run(accuracy,feed_dict={x_input:mnist.test.images,y_label:mnist.test.labels,keep_prob:1})
             print("train accuracy:",train_acc)
             print("test accuracy:",test_acc)
             
             #print("train accuracy:",compute_accuracy(batch_xs, batch_ys,1))
             #print("test accuracy:",compute_accuracy(
                #mnist.test.images, mnist.test.labels,1))
             result = sess.run(merged,feed_dict={x_input: batch_xs, y_label: batch_ys, keep_prob: 0.5})
             file_writer.add_summary(result,i)
             #result = sess.run(merged,feed_dict={x_input: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1})
             #file_writer.add_summary(result,i)
    saver.save(sess,"./my_net/my_CNN_model_final.ckpt")

file_writer.close()
        