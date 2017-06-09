#!/usr/bin/env python
# --*-- coding:utf-8 --*--
import tensorflow as tf
import numpy as np
import os
from sklearn import metrics
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
X_test = np.array(x1)
y_test = np.array(y1)
print "x_test,y_test",X_test,y_test
with tf.Session() as sess:
	#Load the graph
	graph_file = os.path.join(checkpoint_dir, "graph.pb")
	with tf.gfile.FastGFile(graph_file, "rb") as f:
        	graph_def = tf.GraphDef()
        	graph_def.ParseFromString(f.read())
        	tf.import_graph_def(graph_def, name="")
	print(graph_def.ListFields())
	#Get the model's variables
	W = sess.graph.get_tensor_by_name("model/W:0")
	b = sess.graph.get_tensor_by_name("model/b:0")
	# Load the saved variables from the checkpoint back into the session.
	checkpoint_file = os.path.join(checkpoint_dir, "model")
	saver = tf.train.Saver([W, b])
	saver.restore(sess, checkpoint_file)
	# Get the placeholders and the accuracy operation, so that we can compute
	# the accuracy (% correct) of the test set.
    	x = sess.graph.get_tensor_by_name("inputs/x-input:0")
    	y = sess.graph.get_tensor_by_name("inputs/y-input:0")
    	accuracy = sess.graph.get_tensor_by_name("score/accuracy:0")
    	print("Test set accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
	# Also show some other reports.
	inference = sess.graph.get_tensor_by_name("inference/inference:0")
	predictions = sess.run(inference, feed_dict={x: X_test})
	print("\nClassification report:")
	print(predictions)
	#print(metrics.classification_report(y_test.ravel(), predictions))
 	#print("Confusion matrix:")
	#print(metrics.confusion_matrix(y_test.ravel(), predictions))
