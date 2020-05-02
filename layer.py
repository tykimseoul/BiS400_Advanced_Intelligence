import tensorflow as tf


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def convolutional_layer(input,
                        num_input_channels,
                        conv_filter_size,
                        num_filters):
    # define the weights that will be trained using create_weights function
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # create biases using the create_biases function. These are also trained
    biases = create_biases(num_filters)

    # create the convolutional layer
    layer = tf.nn.conv2d(input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # use max-pooling
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    # output is fed to Relu which is the activation function
    layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    input_size = layer.get_shape().as_list()
    # # img_height * img_width * num_channels
    new_size = input_size[-1] * input_size[-2] * input_size[-3]
    # flatten the layer
    return tf.reshape(layer, [-1, new_size])


def fully_connected_layer(input,
                          num_inputs,
                          num_outputs,
                          use_relu=True):
    # trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # takes input x and produces wx+b
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = activation_layer(layer)

    return layer


def activation_layer(layer):
    return tf.nn.relu(layer)
