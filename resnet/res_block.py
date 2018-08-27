# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:44:40 2018

@author: cmy

"""
import tensorflow as tf

def res_block(x_input,kernel_size,in_filter,
              out_filter,stage,training):
    #stage :
    #training：false or ture
    block_name='res_block'+str(stage)
    with tf.name_scope(block_name):
        f1,f2,f3=out_filter
        X_shortcut=x_input
        
        #layer1
        with tf.name_scope(block_name+'layer1'):
            W_conv1=weight_variable([1,1,in_filter,f1])
            X=tf.nn.conv2d(x_input,W_conv1,strides=[1,1,1,1],padding='VALID')
            X=tf.layers.batch_normalization(X,axis=3,training=training)
            X=tf.nn.relu(X)
        
        #layer2
        with tf.name_scope(block_name+'layer2'):
            W_conv2=weight_variable([kernel_size,kernel_size,f1,f2])
            X=tf.nn.conv2d(X,W_conv2,strides=[1,1,1,1],padding='SAME')
            X=tf.layers.batch_normalization(X,axis=3,training=training)
            X=tf.nn.relu(X)
        
        #layer3
        with tf.name_scope(block_name+'layer3'):
            W_conv3=weight_variable([1,1,f2,f3])
            X=tf.nn.conv2d(X,W_conv3,strides=[1,1,1,1],padding='VALID')
            X=tf.layers.batch_normalization(X,axis=3,training=training)
            X=tf.nn.relu(X)
        
        #shuortcut 
        add=tf.add(X,X_shortcut)
        add_reslut=tf.nn.relu(add)
    return add_reslut

def weight_variable(shape,name='W'):
    with tf.name_scope('Weight'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        Weights=tf.Variable(initial)
        tf.summary.histogram(name+'/Weights_function',Weights)
        return Weights


def bias_variable(shape,name):
    with tf.name_scope('Bias'):    
        initial = tf.constant(0.1, shape=shape)
        Biases=tf.Variable(initial)
        tf.summary.histogram(name+'/Biases_function',Biases)
        return Biases
        

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
