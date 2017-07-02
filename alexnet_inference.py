
import tensorflow as tf


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images,dropout):
	parameters = []
	# conv1
	with tf.variable_scope('layer1-conv1'):
		kernel = tf.Variable(tf.truncated_normal([11, 11, 1, 96], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias)
		print_activations(conv1)
		parameters += [kernel, biases]
	
	# lrn1
	with tf.name_scope("layer2-pool1"):
		lrn1 = tf.nn.local_response_normalization(conv1,
											  alpha=1e-4,
											  beta=0.75,
											  depth_radius=2,
											  bias=2.0)
		# pool1
		pool1 = tf.nn.max_pool(lrn1,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1],
						 padding='SAME',
						 name='pool1')
		print_activations(pool1)
		
	# conv2
	with tf.variable_scope('layer1-conv2'):
		kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias)
		print_activations(conv2)
		parameters += [kernel, biases]
	
	# lrn1
	with tf.name_scope("layer2-pool2"):
		lrn2 = tf.nn.local_response_normalization(conv2,
											  alpha=1e-4,
											  beta=0.75,
											  depth_radius=2,
											  bias=2.0)
		# pool1
		pool2 = tf.nn.max_pool(lrn2,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1],
						 padding='SAME',
						 name='pool2')
		print_activations(pool2)	
		
		# conv3
	with tf.variable_scope('layer3-conv3'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(bias)
		print_activations(conv3)
		parameters += [kernel, biases]	
		
	# conv4
	with tf.variable_scope('layer4-conv4'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(bias)
		print_activations(conv4)
		parameters += [kernel, biases]		
		
	# conv5
	with tf.variable_scope('layer5-conv5'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(bias)
		print_activations(conv5)
		parameters += [kernel, biases]	
		
		# pool5
		pool5 = tf.nn.max_pool(conv5,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1],
						 padding='SAME',
						 name='pool5')
		print_activations(pool5)	
		
	with tf.variable_scope('layer6-fc1'):
		
		fc1_weights = tf.get_variable("weight", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc=tf.reshape(pool5,[-1,fc1_weights.get_shape().as_list()[0]])
		fc1_biases= tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')							 
		fc1=tf.add(tf.matmul(fc,fc1_weights),fc1_biases)
		
		fc1=tf.nn.relu(fc1)
		
		#fc1 = tf.nn.relu(tf.matmul(fc, fc1_weights) + fc1_biases)
		print_activations(fc1)
	
	with tf.variable_scope('layer7-fc2'):
		fc2_weights = tf.get_variable("weight", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc2_biases= tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')							 
		fc2=tf.add(tf.matmul(fc1,fc2_weights),fc2_biases)
		fc2=tf.nn.relu(fc2)
		print_activations(fc2)	
		#dropout
		fc2=tf.nn.dropout(fc2,dropout)
		
	with tf.variable_scope('layer8-out'):	
		#輸出層
		out_weights = tf.get_variable("weight", [4096, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
		out_biases= tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
		out=tf.add(tf.matmul(fc2,out_weights),out_biases)
		
	return out
		
