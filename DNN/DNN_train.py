# -*- coding: utf-8 -*-
"""
Created on 12nd June, 2018

@author: cmy

coded by myself 
This is a plain deep neural network, the dataset is mnist,
and visible on tensorboard
The file is used to train the model and save the parameter in 'my_net'
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_train"
logdir = "{}/run-{}/".format(root_logdir, now)


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
n_epochs = 2000
batch_size = 50

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
    
    #defined loss
    with tf.name_scope("loss"):
        xentropy =tf.square(outputs-outputs_pred)# tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outputs, logits=outputs_pred)
        loss = tf.reduce_mean(xentropy, name="loss")
        d=tf.summary.scalar('loss_function',loss)
    
    
    learning_rate = 1.2
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope("eval"):
        correct = tf.equal(tf.argmax(outputs_pred,1), tf.argmax(outputs,1))
        #correct = tf.nn.in_top_k(outputs_pred, tf.to_int64(outputs), 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        e=tf.summary.scalar('accuracy_function',accuracy)
        
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    scope="layer[12]")
    reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
   

with tf.Session(graph=g) as sess:
    #with tf.Session(graph=g) as sess:
    sess.run(init)
    file_writer = tf.summary.FileWriter(logdir, sess.graph) 
    #original_saver.restore( sess,"./trainedSaver/my_model_final.ckpt")
    for epoch in range(n_epochs+1):
        X_batch=mnist.train.images
        y_batch=mnist.train.labels
        X_batch, y_batch = mnist.train.next_batch(100)
        sess.run(training_op, feed_dict={inputs: X_batch, outputs: y_batch})
        
        if epoch %20==0:
            print('training loss:',sess.run(loss,feed_dict={inputs: X_batch, outputs: y_batch}))
            acc_train = sess.run(accuracy,feed_dict={inputs: X_batch, outputs: y_batch})
            acc_test=sess.run(accuracy,feed_dict={inputs:mnist.test.images,outputs:mnist.test.labels})
            print("Train accuracy:",acc_train)
            print("Test accuracy:", acc_test)
            result = sess.run(merged,feed_dict={inputs: X_batch, outputs: y_batch})
            file_writer.add_summary(result,epoch)
            
            #result = sess.run(merged,feed_dict={inputs: X_batch, outputs: y_batch})
            #file_writer.add_summary(result,epoch)
            
    save_path = saver.save(sess, "./my_net/my_model_final.ckpt")
    
file_writer.close() 
