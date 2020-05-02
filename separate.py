import numpy as np
import tensorflow as tf
import os

clusters_n = 2
iteration_n = 1000

dir_path = os.path.dirname(os.path.realpath(__file__))
points = tf.constant(np.load(dir_path + "/data_for_problem2/problem2_training_data.npy"))
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))

distances = tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(points, 0), tf.expand_dims(centroids, 1))), 2)
assignments = tf.argmin(distances, 0)

means = []
for c in range(clusters_n):
    means.append(tf.reduce_mean(
        tf.gather(points,
                  tf.reshape(
                      tf.where(
                          tf.equal(assignments, c)
                      ), [1, -1])
                  ), reduction_indices=[1]))

updated_centroids = tf.assign(centroids, tf.concat(means, 0))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(iteration_n):
    [_, centroid_values, points_values, assignment_values] = sess.run([updated_centroids, centroids, points, assignments])

mapped = list(zip(assignment_values, points_values))

resultFile = open("separation_result.txt", "w")
resultFile.write('centroid 0: {}, centroid 1: {}\n'.format(centroid_values[0], centroid_values[1]))
for (cluster, point) in mapped:
    resultFile.write("cluster {}: {}\n".format(cluster, point))
print("centroids", centroid_values)
print(np.bincount(assignment_values))
