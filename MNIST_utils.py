import struct
from array import array
import matplotlib.pyplot as plt
import numpy as np
import torch

# TODO: these numpy arrays need to be tensors ...
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# Show example images
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(28, 28))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1

def get_images_for_class(images, indices):
    return np.array([images[i] for i in indices])

def get_indices_for_class(class_, loa, n=5):
    """Scan through the label file until we have the required number of images for the requested class."""
    images_ = [i for i, x in enumerate(loa) if x == class_]
    return images_[:n]

def get_images_for_given_class(class_, list_of_labels, list_of_images, n=5):
    indices = get_indices_for_class(class_=class_, loa=list_of_labels, n=n)
    return get_images_for_class(list_of_images, indices)

def build_images_tensor(num_of_classes, number_of_images, list_of_labels, list_of_images):
    return torch.tensor(np.array([get_images_for_given_class(class_=i,
                                                    list_of_labels=list_of_labels,
                                                    list_of_images=list_of_images,
                                                    n=number_of_images)
                          for i in range(num_of_classes)]))
