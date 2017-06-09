#!/usr/bin/env python
# --*-- coding:utf-8 --*--
import tensorflow as tf
import numpy as np
import os

checkpoint_dir = "/home/dl/models/lr_model"
#Read x and y
def loadDataSet():
	data = []; label = []
	fr = open("/home/dl/models/data.txt")
	for line in fr.readlines():
		lineArr = line.strip().split()
		data.append([float(lineArr[0]),float(lineArr[1]),float(lineArr[2])])
		label.append([float(lineArr[3])])
	return data, label

x1, y1 = loadDataSet()
x_train = np.array(x1)
y_train = np.array(y1)
print "x_data,y_data",x_train,y_train

with tf.name_scope("model"):
	W = tf.Variable(tf.zeros([3,1]),name="W")
	b = tf.Variable(tf.zeros([1,1]),name="b")

with tf.name_scope("inputs"):
	x = tf.placeholder(tf.float32, [None, 3], name="x-input")
	y = tf.placeholder(tf.float32, [None, 1], name="y-input")

y_pred = tf.sigmoid(tf.matmul(x,W)+b)
with tf.name_scope("hyperparmeters"):
	regularization = tf.placeholder(tf.float32,name="regularization")
	learning_rate = tf.placeholder(tf.float32,name="learning_rate")
with tf.name_scope("loss-function"):
	loss = tf.losses.log_loss(labels=y,predictions=y_pred)
	loss += regularization * tf.nn.l2_loss(W)
with tf.name_scope("train"):
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
with tf.name_scope("inference"):
	inference = tf.to_float(y_pred>0.5,name="inference")
with tf.name_scope("score"):
    correct_prediction = tf.equal(inference, y)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name="accuracy")
saver = tf.train.Saver()
with tf.Session() as sess:
	tf.train.write_graph(sess.graph_def, checkpoint_dir, "graph.pb", False)
	init = tf.global_variables_initializer()
	sess.run(init)
	loss_value = sess.run(loss, feed_dict={x: x_train, y: y_train, regularization: 0})
	print("Initial loss:", loss_value)
	feed = {x: x_train, y: y_train, learning_rate: 1e-2, regularization: 1e-5}
	for step in range(100):
		sess.run(train_op,feed_dict=feed)
		if step %10 ==0:
			train_accuracy, loss_value = sess.run([accuracy, loss],feed_dict=feed)
			checkpoint_file = os.path.join(checkpoint_dir, "model")
			saver.save(sess, checkpoint_file)
			print("*** SAVED MODEL ***")
			print("step: %4d, loss: %.4f, training accuracy: %.4f" %(step, loss_value, train_accuracy))
