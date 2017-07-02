import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cv2,csv
import alexnet_inference2

def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
	return onehot

def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`"""
	if kind=='train':
		labels_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\train-labels.idx1-ubyte')		
		images_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\train-images.idx3-ubyte')
	else:
		labels_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\t10k-labels.idx1-ubyte')		
		images_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\t10k-images.idx3-ubyte')
	
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II',
								 lbpath.read(8))
		labels = np.fromfile(lbpath,
							 dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII",
											   imgpath.read(16))
		images = np.fromfile(imgpath,
							 dtype=np.uint8).reshape(len(labels), 784)

	return images, labels

# Parameters
MODEL_SAVE_PATH = "./alexnet/"
MODEL_NAME = "alexnet_model"
learning_rate = 0.001
BATCH_SIZE = 100
display_step = 10
TRAINING_STEPS=1500
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5# Dropout, probability to keep units

def train(X_train,y_train_lable):
	shuffle=True
	batch_idx=0
	batch_len =int( X_train.shape[0]/BATCH_SIZE)
	train_acc=[]
	train_idx=np.random.permutation(batch_len)#打散btach_len=600 group

	# tf Graph input
	x_ = tf.placeholder(tf.float32, [None, n_input])	
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	x = tf.reshape(x_, shape=[-1, 28, 28, 1])

	# Construct model
	pred = alexnet_inference2.inference(x,  keep_prob)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# 初始化TensorFlow持久化類。
	saver = tf.train.Saver()
	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		print ("Start  training!")
		# Keep training until reach max iterations:
		while step	< TRAINING_STEPS:
			#batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
			if shuffle==True:
				batch_shuffle_idx=train_idx[batch_idx]
				batch_xs=X_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
				batch_ys=y_train_lable[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]	
			else:
				batch_xs=X_train[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE]
				batch_ys=y_train_lable[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE]
		
			if batch_idx<batch_len:
				batch_idx+=1
				if batch_idx==batch_len:
					batch_idx=0
			else:
				batch_idx=0
			reshaped_xs = np.reshape(batch_xs, (
					BATCH_SIZE,
					28,
					28,
					1))
			
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: reshaped_xs, y: batch_ys,
										keep_prob: dropout})
			# Calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x: reshaped_xs,
																y: batch_ys,
																keep_prob: 1.})
			train_acc.append(acc)
			if step % display_step == 0:
				print("Step: " + str(step) +" Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + \
					"{:.6f}".format(loss) + ", Training Accuracy= " + \
					"{:.5f}".format(acc))
			step += 1
		print("Optimization Finished!")
		print("Save model...")
		#saver.save(sess, "./alexnet/alexnet_model")
		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
							
										
def main(argv=None):

	#mnist = input_data.read_data_sets("./mnist", one_hot=True)
	#### Loading the data
	X_train, y_train = load_mnist('..\mnist', kind='train')
	print('X_train Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1])) #X_train=60000x784
	#X_test, y_test = load_mnist('mnist', kind='t10k')					 #X_test=10000x784
	#print('X_test Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
	print("train_data.shape=",X_train.shape)
	print("train_label.shape=",y_train.shape)
	
	mms=MinMaxScaler()
	X_train=mms.fit_transform(X_train)

	#stdsc=StandardScaler()
	#X_train=stdsc.fit_transform(X_train)
	#X_test=stdsc.transform(X_test)

	y_train_lable = encode_labels(y_train,10)
	print("y_train_lable.shape=",y_train_lable.shape)
	##============================
	
	train(X_train,y_train_lable)

if __name__ == '__main__':
	main()

