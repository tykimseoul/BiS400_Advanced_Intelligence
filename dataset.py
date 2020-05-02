import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


class DataSet(object):
    def __init__(self, images, labels, img_names, cls):
        self.num_examples = images.shape[0]
        print(images.shape)
        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.cls = cls
        self.epochs_done = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            # after each epoch, update this
            self.epochs_done += 1
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch

        return self.images[start:end], self.labels[start:end], self.img_names[start:end], self.cls[start:end]


def load_train(train_path, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    for fields in classes:
        index = classes.index(fields)
        print('Reading {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            # image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


def read_train_sets(train_path, classes, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets
