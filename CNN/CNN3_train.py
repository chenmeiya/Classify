# -*- coding: utf-8 -*-
"""
Created on 12nd June, 2018

@author: CMY
"""

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_train"
logdir = "{}/run-{}/".format(root_logdir, now)
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''def compute_accuracy(v_xs, v_ys,prob):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: prob})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    a=tf.summary.scalar('accuracy_function',accuracy)
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: prob})
    
    return result'''

def weight_variable(shape,name):
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

# define placeholder for inputs to network
with tf.Graph().as_default() as g:
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 784],name='input') # 28x28
        ys = tf.placeholder(tf.float32, [None, 10],name='label')
        keep_prob = tf.placeholder(tf.float32,name='prob')
        x_image = tf.reshape(xs, [-1, 28, 28, 1],name='image')
        # print(x_image.shape)  # [n_samples, 28,28,1]
    
    ## conv1 layer ##
    with tf.name_scope('layer1'):
        W_conv1 = weight_variable([5,5, 1,32],name='layer1') # patch 5x5, in size 1, out size 32
        b_conv1 = bias_variable([32],name='layer1')
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
        h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32
    
    ## conv2 layer ##
    with tf.name_scope('layer2'):
        W_conv2 = weight_variable([5,5, 32, 64],name='layer2') # patch 5x5, in size 32, out size 64
        b_conv2 = bias_variable([64],name='layer2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
        h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64
    
    ## func1 layer ##
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7*7*64, 1024],name='fc1')
        b_fc1 = bias_variable([1024],name='fc1')
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        h_fc1_drop=h_fc1
   
    with tf.name_scope('fc2'):
        ## func2 layer ##
        W_fc2 = weight_variable([1024, 10],name='fc2')
        b_fc2 = bias_variable([10],name='fc2')
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        
    with tf.name_scope('loss'):
    # the error between prediction and real data
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
        tf.summary.scalar('loss_function',cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
    with tf.name_scope('eval'):
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy_function',accuracy)
    
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    
    
with tf.Session(graph=g) as sess:       
        # important step
    sess.run(init)
    file_writer = tf.summary.FileWriter(logdir, sess.graph) 
    for i in range(2000+1):        
        X_batch=mnist.train.images
        Y_batch=mnist.train.labels
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_loss=sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
       
        if i % 50 == 0:
             train_acc=sess.run(accuracy,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
             test_acc=sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1})
             print("train accuracy:",train_acc)
             print("test accuracy:",test_acc)
             
             #print("train accuracy:",compute_accuracy(batch_xs, batch_ys,1))
             #print("test accuracy:",compute_accuracy(
                #mnist.test.images, mnist.test.labels,1))
             result = sess.run(merged,feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
             file_writer.add_summary(result,i)
             #result = sess.run(merged,feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1})
             #file_writer.add_summary(result,i)
    saver.save(sess,"./my_net/my_CNN_model_final.ckpt")

file_writer.close()