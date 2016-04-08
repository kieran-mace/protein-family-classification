# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.astype(int)] = 1
  return labels_one_hot


data = np.genfromtxt('data/query_db.out',filling_values=30)
train_dataset = data[:,1::]
train_labels = dense_to_one_hot(data[:,0],num_classes=8)

num_labels = 8
num_channels = 1 # grayscale

# def reformat(dataset, labels):
#   dataset = dataset.astype(np.float32)
#   labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
#   return dataset, labels
# train_dataset, train_labels = reformat(train_dataset, train_labels)
print('Training set', train_dataset.shape, train_labels.shape)




def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 10
input_layer = 4854
middle_layer1 = 100
middle_layer2 = 10
middle_layer3 = 20
output_layer = num_labels
graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, input_layer))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  #tf_valid_dataset = tf.constant(valid_dataset)
  #tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([input_layer, middle_layer1]))
  biases1 = tf.Variable(tf.zeros([middle_layer1]))
  weights2 = tf.Variable(
    tf.truncated_normal([middle_layer1, middle_layer2]))
  biases2 = tf.Variable(tf.zeros([middle_layer2]))
  weights3 = tf.Variable(
    tf.truncated_normal([middle_layer2, middle_layer3]))
  biases3 = tf.Variable(tf.zeros([middle_layer3]))
  weights4 = tf.Variable(
    tf.truncated_normal([middle_layer3, output_layer]))
  biases4 = tf.Variable(tf.zeros([output_layer]))
  # Model.
  def model(x0):
    x1 = tf.nn.relu(tf.matmul(x0, weights1) + biases1)
    x2 = tf.nn.relu(tf.matmul(x1, weights2) + biases2)
    x3 = tf.nn.relu(tf.matmul(x2, weights3) + biases3)
    y = tf.nn.softmax(tf.matmul(x3, weights4) + biases4)
    return y
  # Training computation.
  logits = model(tf_train_dataset)
  losses = [tf.nn.l2_loss(w) for w in [weights1,weights2,weights3,weights4]]
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + 0.1 * tf.add_n(losses)

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#   global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
#   starter_learning_rate = 0.1
#   learning_rate = tf.train.exponential_decay(0.5, global_step, 100000, 0.96, staircase=True)
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  #valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  #test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 2001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 200 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      #print("Validation accuracy: %.1f%%" % accuracy(
      #  valid_prediction.eval(), valid_labels))
  #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
