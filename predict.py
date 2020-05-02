import tensorflow as tf
import numpy as np
import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + '/data_for_problem1_smallsize/test_dataset_unlabeled/'
image_size = 200
num_channels = 3


def predict_single(name):
    images = []
    image = cv2.imread(filename + name)
    # image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    # feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result
    print_result(name, result[0])


def print_result(name, result):
    if result[0] > result[1]:
        print("{0}: 1\tAbnormal: {1:.5f}".format(name.split(".")[0], result[0]))
    else:
        print("{0}: 0\tNormal: {1:.5f}".format(name.split(".")[0], result[1]))


# restore the saved model
sess = tf.Session()
saver = tf.train.import_meta_graph('xray_model_22.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

files = os.listdir(filename)
if '.DS_Store' in files:
    files.remove('.DS_Store')
files.sort(key=lambda name: int(name.partition('.')[0]))
for file in files:
    # print(file)
    predict_single(file)
