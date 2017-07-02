import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import alexnet_inference2
import alexnet_train2
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def evaluate(X_test,y_test_lable):
	with tf.Graph().as_default() as g:
	
		# 定義輸出為4維矩陣的placeholder
		x_ = tf.placeholder(tf.float32, [None, alexnet_train2.n_input])	
		x = tf.reshape(x_, shape=[-1, 28, 28, 1])
		y = tf.placeholder(tf.float32, [None, alexnet_train2.n_classes])
	
		# Construct model
		pred = alexnet_inference2.inference(x, 1)     #dropout=1

		# Evaluate model
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
		test_batch_len =int( X_test.shape[0]/alexnet_train2.BATCH_SIZE)
		test_acc=[]
		
		test_xs = np.reshape(X_test, (
					X_test.shape[0],
					28,
					28,
					1))
		
		batchsize=alexnet_train2.BATCH_SIZE
	
		# 'Saver' op to save and restore all the variables
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess,"./alexnet/alexnet_model")

			for i in range(test_batch_len):
				temp_acc= sess.run(accuracy, feed_dict={x: test_xs[batchsize*i:batchsize*i+batchsize], y: y_test_lable[batchsize*i:batchsize*i+batchsize]})
				test_acc.append(temp_acc)
				print ("Test  batch ",i,":Testing Accuracy:",temp_acc)	

			t_acc=tf.reduce_mean(tf.cast(test_acc, tf.float32))	
			print("Average Testing Accuracy=",sess.run(t_acc))
			return

def main(argv=None):
	#### Loading the data
	#X_train, y_train = alexnet_train2.load_mnist('..\mnist', kind='train')
	#print('X_train Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1])) #X_train=60000x784
	X_test, y_test = alexnet_train2.load_mnist('mnist', kind='t10k')					 #X_test=10000x784
	print('X_test Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
	mms=MinMaxScaler()
	#X_train=mms.fit_transform(X_train)
	X_test=mms.fit_transform(X_test)

	#y_train_lable = alexnet_train2.encode_labels(y_train,10)
	y_test_lable = alexnet_train2.encode_labels(y_test,10)
	##============================
	
	evaluate(X_test,y_test_lable)

if __name__ == '__main__':
	main()
