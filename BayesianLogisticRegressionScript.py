#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Trains a Bayesian logistic regression model on synthetic data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import flags
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import roc_curve

tfd = tfp.distributions

flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=1500,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=32,
                     help="Batch size.")
#flags.DEFINE_string(
    #"model_dir",
    #default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         #"logistic_regression/"),
    #help="Directory to put the model's fit.")
#flags.DEFINE_integer("num_examples",
                     # help="Number of datapoints to generate.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                    help="Monte Carlo samples to visualize weight posterior.")

FLAGS = flags.FLAGS
#TRAINING_PERCENTAGE = 0.8


# In[ ]:


#tf.reset_default_graph()
features= [ 'type', 'sender', 'messageID', 'pos/2','spd/0', 'spd/1', 'pos/2']
targets= ['class']


# In[ ]:



def main(argv):
    del argv  # unused
    if tf.gfile.Exists(FLAGS.model_dir):
        tf.logging.warning(
            "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
        tf.gfile.MakeDirs(FLAGS.model_dir)


#Building an iterator over training batches 

PATH = os.getcwd()
preprocessed_data_path = PATH+'/data1.csv'

#Building an iterator over training batches 
def build_input_pipeline(preprocessed_data_path,batch_size):
    np.save('work',preprocessed_data_path)
    with np.load('work.npy')as loaded_data:
        loaded_data = loaded_data.astype(np.float32)
        print(loaded_data)
        features = loaded_data.drop(['class'],axis =1, inplace =True)
        labels = loaded_data['class']
        features = features.astype(np.float32)
        labels = labels.astype(np.float32)


""""def build_input_pipeline(preprocessed_data_path, batch_size):
    with np.load(preprocessed_data_path) as loaded_data:
        features = loaded_data.drop(['class'],axis =1, inplace =True)
        labels = data['class']
        features = features.astype(np.float32)
        labels = labels.astype(np.float32)""""
#Splitting into training, validation and testing sets
random.shuffle(features)
split_1 = int(0.8*len(features))
split_2 = int(0.9*len(features))
training_features = features[:split_1]
validation_features = features[split_1:split_2]
testing_features = features[split_2:]
training_labels = labels[:split_1]
validation_labels = labels[split_1:split_2]
testing_labels = labels[split_2]
# Z-normalising: (note with respect to training data)
training_features = (training_features -np.mean(training_features, axis =0))/np.std(training_features, axis =0)
validation_features = (validation_features-np.mean(training_features, axis=0))/np.std(validation_features, axis=0)
testing_featues = (testing_features-np.mean(training_features, axis=0))/np.std(validation_features,axis=0)

#Creating the tf.Dataset object 
training_dataset = tf.data.Dataset.from_tensor_slices((training_features, training_labels))
print(training_dataset)

#Shuffle the dataset (note shuffle argument much larger than training size)
# and form a batch of size batch_size
training_batches = training_dataset.shuffle(2000).repeat().batch(batch_size)
training_iterator = training_batches.make_one_shot_iterator()

#Building a iterator over the heldout set with batch_size = heldout_size,
#i.e., return the entire heldout set as a constant
heldout_dataset = tf.data.Dataset.from_tensor_slices(
(validation_features, validation_labels))
heldout_frozen = (heldout_dataset.take(len(validation_features)).
                  repeat().batch(len(validation_features)))
heldout_iterator = heldout_frozen.make_one_shot_iterator()

#Building a iterator over the test set with batch_size = heldout_size,
#i.e., return the entire test set as a constant
test_dataset = tf.data.Dataset.from_tensor_slices(test_feature, test_labels)
test_iterator = test_dataset.make_one_shot_iterator()

#Combine these into a feedable iterator that can switch between training
# and validation and testing inputs 
# Here should the minibatch increment be defined 
handle = tf.placeholder(tf.string, shape=[],)
feedable_iterator = tf.data.Iterator.from_string_handle(
handle, training_batches.output_types, training_batches.output_shapes)
features_final, labels_final = feedable_iterator.get_next()
return features_final, labels_final, handle, training_iterator, heldout_iterator, test_iterator


# In[ ]:

with tf.name_scope("logistic_regression", values=[features]):
    layer = tfp.layers.DenseFlipout(
    units=1,
    activation=None,
    kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
    bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())
    logits = layer(features)
    labels_distribution = tfd.Bernoulli(logits=logits)

  # Compute the -ELBO as the loss, averaged over the batch size.
neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
kl = sum(layer.losses) / FLAGS.batch_size
elbo_loss = neg_log_likelihood + kl

predictions = tf.cast(logits > 0, dtype=tf.int32)
accuracy, accuracy_update_op = tf.metrics.accuracy(
labels=labels, predictions=predictions)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

with tf.Session() as sess:
    
    sess.run(init_op)
    
    #Run the training loop
    train_handle = sess.run(training_iterator.string_handle())
    heldout_handle = sess.run(heldout_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())


    # Fit the model to data.
    for step in range(FLAGS.max_steps):
      _ = sess.run([train_op, accuracy_update_op],feed_dict={handle:train_handle})
    
      if step % 100 == 0:
        loss_value, accuracy_value = sess.run([elbo_loss, accuracy],feed_dict={handle:train_handle})
        loss_value_validation, accuracy_value_validation = sess.run([elbo_loss, accuracy]
                                                                    feed_dict = {handle:heldout_handle})
        loss_value_test, accuracy_value_test = sess.run([elbo_loss, accuracy]
                                                                    feed_dict = {handle:test_handle})
        print("Step: {:>3d} Loss: [{:.3f}, {:.3f},{:.3f}]Accuracy: [{:.3f}, {:.3f},{:.3f}]".format(
            step, loss_value,loss_value_validation,loss_value_test, accuracy_value,
            accuracy_value_validation,accuracy_value_test ))

    # Visualize some draws from the weights posterior.
    w_draw = layer.kernel_posterior.sample()
    b_draw = layer.bias_posterior.sample()
    candidate_w_bs = []
    for _ in range(FLAGS.num_monte_carlo):
        w, b = sess.run((w_draw, b_draw))
        candidate_w_bs.append((w, b))


  def generate_ROC_curve():
    with tf.Session() as sess:
       logits = sess.run([logits], feed_dict = {handle:test_handle})

    roc_curve(np.array(test_labels), logits) # Logits, a tensor resulting from a graph evaluation, is a NumPy array


if __name__ == "__main__":
  tf.app.run()


# In[ ]:





# In[ ]:




