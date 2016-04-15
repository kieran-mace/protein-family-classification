from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import range

# settings
train = 0.85
valid = 0.15
test = 0.00

dropout_prob = 0.99

x_size = 4854
y_size = 92

batch_size = 200

steps = 51
# Import data

data = np.genfromtxt('data/silver_standard_all_matrix_withNA.txt',filling_values=-30.0).astype(np.float32)

shuffle = np.random.choice(data.shape[0],size=data.shape[0],replace=False)
data = data[shuffle,:]



train_ind = range(0                                     ,int(round((data.shape[0]*train))))
valid_ind = range(int(round(data.shape[0]*train))       ,int(round(data.shape[0]*(1.0-test))))
test_ind =  range(int(round(data.shape[0]*(1.0-test)))  ,data.shape[0])

train_dataset = data[train_ind,1::]
train_labels = data[train_ind,0].astype(int) #dense_to_one_hot(data[10::,0],num_classes=8)

valid_dataset = data[valid_ind,1::]
valid_labels = data[valid_ind,0].astype(int) #dense_to_one_hot(data[0:10,0],num_classes=8)

test_dataset = data[test_ind,1::]
test_labels = data[test_ind,0].astype(int) #dense_to_one_hot(data[0:10,0],num_classes=8)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read, and
  adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

# Create a multilayer model.
graph = tf.Graph()
with graph.as_default():
  # Input placehoolders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, x_size], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)

  hidden1 = nn_layer(x, x_size, 500, 'layer1')
  dropped = tf.nn.dropout(hidden1, keep_prob)
  y = nn_layer(dropped, 500, y_size, 'layer2', act=tf.nn.relu)


  with tf.name_scope('cross_entropy'):
    diff = tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(
        0.0001).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
      #correct_prediction = tf.nn.in_top_k(y, y_, 3)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

with tf.Session(graph=graph) as sess:
  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter('./train', sess.graph)
  test_writer = tf.train.SummaryWriter('./test', sess.graph)
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      ind = np.random.choice(train_dataset.shape[0],
                             size=train_dataset.shape[0],
                             replace=False)
      xs, ys = train_dataset[ind,:], train_labels[ind]
      k = dropout_prob
    else:
      xs, ys = valid_dataset, valid_labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(steps):
    if i % 50 == 0:  # Record summaries and test-set accuracy
      summary, acc, ce = sess.run([merged, accuracy, cross_entropy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
      print('Cross Entropy at step %s: %s' % (i, ce))
    else: # Record train set summarieis, and train
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)
