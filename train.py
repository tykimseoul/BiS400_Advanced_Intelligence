import dataset
import tensorflow as tf
import os
import layer as layer

from numpy.random import seed
from tensorflow import set_random_seed

seed(1)
set_random_seed(2)

# 20% of the data used for validation
validation_size = 0.2
img_size = 200
num_channels = 3

# Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 50

filter_size_conv2 = 3
num_filters_conv2 = 50

filter_size_conv3 = 3
num_filters_conv3 = 100

fc_layer_size = 200
batch_size = 32

# input data
dir_path = os.path.dirname(os.path.realpath(__file__))
classes = os.listdir(dir_path + '/data_for_problem1_smallsize/training_data')
if '.DS_Store' in classes:
    classes.remove('.DS_Store')
num_classes = len(classes)
assert num_classes == 2, "Make sure there are only two datasets!"

train_path = 'data_for_problem1_smallsize/training_data'

# load all the training and validation images and labels
data = dataset.read_train_sets(train_path, classes, validation_size=validation_size)

print("Training set:\t{} files".format(len(data.train.labels)))
print("Validation set:\t{} files".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1 = layer.convolutional_layer(x,
                                        num_input_channels=num_channels,
                                        conv_filter_size=filter_size_conv1,
                                        num_filters=num_filters_conv1)

layer_conv2 = layer.convolutional_layer(layer_conv1,
                                        num_input_channels=num_filters_conv1,
                                        conv_filter_size=filter_size_conv2,
                                        num_filters=num_filters_conv2)

layer_conv3 = layer.convolutional_layer(layer_conv2,
                                        num_input_channels=num_filters_conv2,
                                        conv_filter_size=filter_size_conv3,
                                        num_filters=num_filters_conv3)

layer_flat = layer.flatten_layer(layer_conv3)

layer_fc1 = layer.fully_connected_layer(layer_flat,
                                        num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                        num_outputs=fc_layer_size,
                                        use_relu=True)

layer_fc2 = layer.fully_connected_layer(layer_fc1,
                                        num_inputs=fc_layer_size,
                                        num_outputs=num_classes,
                                        use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
saver = tf.train.Saver()


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


def train(epochs):
    for i in range(epochs * int(data.train.num_examples / batch_size)):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './xray_model_{}'.format(epochs))


train(22)
