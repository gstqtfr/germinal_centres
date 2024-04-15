import torch
from itertools import count
from functools import reduce, partial
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(30, 15))
    fig.tight_layout()
    for ind, title in enumerate(figures):
        # ... uncomment this if we want moody, noir images
        # axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].imshow(figures[title])
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional

def output_threshold_figures(_labels, _images, _threshold, N=32):
    l = []

    for _class in range(10):
        l.append(get_total_frequencies_per_class(_labels=_labels, _images=_images, _class=_class, _threshold=_threshold, N=N))

    figures = {'image of '+str(i): l[i] for i in range(10)}
    plot_figures(figures, 5, 2)

    return l


def split_batch(target_array, _class):
    """Split the array by class

    Parameters
    ----------
    target_array: dataset containing labels
    _class: the specific label we're looking for - e.g. 7 in MNIST dataset, etc.
    returns: list of indices into the image dataset
    """
    return [ind for ind, ele in zip(count(), target_array) if ele == _class]


def batch_threshold_count(matrix_batch, f):
    """Count the number of elements in the matrix which are above a certain threshold t
    matrix_batch: a batch's-worth of matrices to be processed
    f: threshold function: partial funciton with the threshold already bound. if above the threshold, 1, else 0.
    returns: cumulative frequency table of the results
    """
    # zero_threshold_tensor = torch.tensor(zero_threshold_array)
    # m2 = torch.where(m1 >= t, 1, 0)
    # a problem here would be that we might need to change the threshold; this can be done by
    # creating a partial function with the threshold already set
    return torch.tensor(np.array(list(map(f, matrix_batch))))


def create_batch(data, N=32):
    m = []
    for idx in range(N):
        m.append(torch.from_numpy(np.array(data[idx])))
    return torch.stack(m)


def get_batch_images(labels, images, _class, start=0, stop=32):
    # this gets the labels for the specified class
    # returns the indices into the images array
    indices = split_batch(target_array=labels, _class=_class)[start:stop]
    # JKK: add create_batch here, for convenience?
    return [images[idx] for idx in indices]

def get_random_batch(labels, images, N=32):
    """Gets a random sample of the indices from the label dataset of size N."""
    #
    indices = torch.randint(high=len(labels), size=(N,))
    # JKK: add create_batch here, for convenience?
    # return [images[idx] for idx in indices]

    image_batch = [images[idx] for idx in indices]
    label_batch = [labels[idx] for idx in indices]

    return torch.from_numpy(np.array(image_batch)), torch.from_numpy(np.array(label_batch))

def get_binary_threshold(m, threshold=127):
    return torch.where(m >= threshold, 1, 0)


def get_number_of_batches(dataset, _class, batch_size):
    return math.floor(len(split_batch(dataset, _class)) / batch_size)

def scs(receptor_, img_, p=2.5, eps=1e-06):
    # we need to preserve the sign of the receptor
    signs = torch.sign(receptor_)
    # more efficient to do this outside & pass it as a ref to func, but we might
    # want to reduce this as we go along ...
    cos = torch.nn.CosineSimilarity(eps=eps)
    similarity = cos(torch.abs(receptor_).type(torch.float), img_.type(torch.float))
    return signs * torch.pow(similarity, p)

def get_sharpened_cosine_similarity_of_naive_max_classifiers(list_of_classifiers, image, p=2.5):

    classifications = torch.zeros(len(list_of_classifiers))

    for idx, maxc in enumerate(list_of_classifiers):
        sharpened_cosine_similarity = scs(receptor_=maxc, img_= image, p=p)
        sum_similarity = torch.sum(sharpened_cosine_similarity)
        #print(f"{idx}: {sum_similarity}")
        classifications[idx] = sum_similarity

    return classifications


def get_frequencies_per_class(_labels, _images, _class, _threshold, N=32):
    """Gets a frequency table of the occurrence of values above a given threshold."""
    discriminant = partial(get_binary_threshold, threshold=_threshold)
    # JKK: need a way to derive this, but can;t be bothered now, need to crack on ...
    accumulated_frequency_table = torch.zeros((28, 28), dtype=torch.float32)
    number_of_batches = get_number_of_batches(dataset=_labels, _class=_class, batch_size=N)

    start = 0
    stop = N

    for batch_idx in range(number_of_batches):
        images_batch_by_class = get_batch_images(labels=_labels, images=_images, _class=_class, start=start, stop=stop)
        batch = create_batch(images_batch_by_class)
        batch_of_images = batch_threshold_count(batch, discriminant)
        frequency_table = reduce(
            torch.Tensor.add_,
            batch_of_images,
            torch.zeros_like(batch_of_images[0])  # optionally set initial element to avoid changing `x`
        )
        start = stop
        stop = stop + N
        # then we need to add to the accumulation table
        accumulated_frequency_table = accumulated_frequency_table + frequency_table

    return accumulated_frequency_table / number_of_batches


def get_total_frequencies_per_class(_labels, _images, _class, _threshold, N=32):
    """Gets a frequency table of the occurrence of values above a given threshold."""
    discriminant = partial(get_binary_threshold, threshold=_threshold)
    # JKK: need a way to derive this, but can;t be bothered now, need to crack on ...
    accumulated_frequency_table = torch.zeros((28, 28), dtype=torch.float32)
    number_of_batches = get_number_of_batches(dataset=_labels, _class=_class, batch_size=N)
    start = 0
    stop = N

    for batch_idx in range(number_of_batches):
        images_batch_by_class = get_batch_images(labels=_labels, images=_images, _class=_class, start=start, stop=stop)
        batch = create_batch(images_batch_by_class)
        batch_of_images = batch_threshold_count(batch, discriminant)
        frequency_table = reduce(
            torch.Tensor.add_,
            batch_of_images,
            torch.zeros_like(batch_of_images[0])  # optionally set initial element to avoid changing `x`
        )
        start = stop
        stop = stop + N
        # then we need to add to the accumulation table
        accumulated_frequency_table = accumulated_frequency_table + frequency_table

    return accumulated_frequency_table