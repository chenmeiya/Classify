
# -*- coding: utf-8 -*-
"""
Created on 12nd June, 2018

@author: cmy

coded by myself 
This is a plain deep neural network, the dataset is mnist
visible on tensorboard
This is the test file, by loading parameters in trained model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#defined the neron layer function
def neron_layer(inputs,n_inputs,n_outputs,name,activation_function=None):
    with tf.name_scope(name):
        #n_inputs=int(inputs.get_shape()[1])     
        stddev = 2 / np.sqrt(n_inputs)#define the deviation
        with tf.name_scope(name+'Weights'):
            Weights=tf.Variable(tf.random_normal([n_inputs,n_outputs],stddev=stddev),dtype=tf.float32,name='W')
            tf.summary.histogram(name+'/Weights_function',Weights)
        with tf.name_scope(name+'Biases'):
            Biases=tf.Variable(tf.random_normal([1,n_outputs]),dtype=tf.float32,name='B')
            tf.summary.histogram(name+'/Biases_function',Biases)
        z=tf.matmul(inputs,Weights)
        z=tf.add(z,Biases)
        if activation_function==None:
            outputs= z
        else:
            outputs=activation_function(z)
        tf.summary.histogram(name+'/outputs',outputs)
        return outputs
#set node's number
n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10

with tf.Graph().as_default()  as g:
    with tf.name_scope('inputs'):
        inputs=tf.placeholder(tf.float32,[None,784],name='inputs_data')
        outputs=tf.placeholder(tf.float32,[None,10],name='outputs')
    
    #layer 1
    hidden1=neron_layer(inputs,n_inputs,n_hidden1,name='layer1',activation_function=tf.nn.relu)
    
    #layer 2
    hidden2=neron_layer(hidden1,n_hidden1,n_hidden2,name='layer2',activation_function=tf.nn.relu)
    
    #layer 3
    outputs_pred=neron_layer(hidden2,n_hidden2,n_outputs,name='outputs',activation_function=tf.sigmoid)
       
    with tf.name_scope("eval"):
        prediction=tf.argmax(outputs_pred,1)
        correct = tf.equal(tf.argmax(outputs_pred,1), tf.argmax(outputs,1))
        #correct = tf.nn.in_top_k(outputs_pred, tf.to_int64(outputs), 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        e=tf.summary.scalar('accuracy_function',accuracy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    scope="layer[12]")
    reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
    #original_saver = tf.train.Saver(reuse_vars_dict) # saver to restore the original model
    
with tf.Session(graph=g) as sess:
    #with tf.Session(graph=g) as sess:
    sess.run(init)
    saver.restore(sess,"./my_net/my_model_final.ckpt")
    index=8
    test_vector=mnist.test.images[index,:].reshape(1,784)
    test_label=mnist.test.labels[index,:].reshape(1,10)
    test_image=test_vector.reshape(28,28)
    acc_prediction=sess.run(prediction,feed_dict={inputs: test_vector,outputs:test_label})
    print("cassification:",acc_prediction)
    acc_test = sess.run(accuracy,feed_dict={inputs: test_vector,outputs:test_label})
    print( "Test accuracy:", acc_test)#epoch, "Train accuracy:",
    ax=plt.figure()
    
    if acc_test==1:
        sen='识别正确'
    else:
        sen='识别错误'
    print('识别结果：',str(acc_prediction),',',str(sen))
    im=np.ones((28,28,3))
    im[:,:,1]=test_image
    im[:,:,2]=test_image
    im[:,:,0]=test_image
    plt.imshow(im)
    plt.title(['识别结果:',str(acc_prediction),str(sen)])
    
    
    