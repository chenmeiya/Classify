# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:27:10 2018

@author: cmy

"""

import tensorflow as tf 
import res_block as res
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']

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
        
        
    with tf.name_scope('eval'):
        pred=tf.argmax(prediction,1)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
       

    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    
with tf.Session(graph=g) as sess:       
    # important step
    sess.run(init)
    index=10 
    X_batch=mnist.test.images[index,:].reshape(1,784)
    Y_batch=mnist.test.labels[index,:].reshape(1,10)
    image=X_batch.reshape(28,28)
    saver.restore(sess,"./my_net/my_CNN_model_final.ckpt")
    test_acc=sess.run(accuracy,feed_dict={x_input:X_batch,y_label:Y_batch,keep_prob:1})
    print("test accuracy:",test_acc)
    result=sess.run(pred,feed_dict={x_input:X_batch,y_label:Y_batch,keep_prob:1})
    
    plt.figure()
    if test_acc==1:
        sen='识别正确'
    else:
        sen='识别错误'
    print('识别结果：',str(result),',',str(sen))
    im=np.ones((28,28,3))
    im[:,:,1]=image
    im[:,:,2]=image
    im[:,:,0]=image
    plt.imshow(im)
    plt.title(['识别结果:',str(result),str(sen)])
    
     
   

        