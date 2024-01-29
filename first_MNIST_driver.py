#!/usr/bin/env python

from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import numpy as np
import torch
import time
import torch_apc_utils as ap
import MNIST_utils as mu
from os.path import join

def string_build_arg_list(repertoire_, device_, neighbourhood_list_, images_for_classes_, R_SZ_, C_SZ_, IMGS_, rho=0.99):
    arguments=[]
    for idx_ in range(R_SZ_):
        arguments.append((repertoire_[idx_], idx_, device_, neighbourhood_list_, images_for_classes_[idx_], C_SZ_, rho))
    return arguments

def main():

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    ITERATIONS = 50

    apc_shape = (28, 28)
    R_SZ = 10
    C_SZ = 20
    rep_shape = (R_SZ, *apc_shape)
    # number of images per class
    IMGS = 5
    # number of classes
    NC=10

    # Set file paths based on added MNIST Datasets

    datasets_path = '/home/johnny/Documents/datasets/MNIST'

    training_images_filepath = join(datasets_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(datasets_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(datasets_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(datasets_path, 't10k-labels.idx1-ubyte')

    # Load MINST dataset
    print('Loading MNIST dataset...')
    mnist_dataloader = mu.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                          test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    print('MNIST dataset loaded.')

    # TODO: wrap in np.array
    # images_for_classes = {i: np.array(mu.get_images_for_given_class(class_=i,
    #                                                                 list_of_labels=y_train,
    #                                                                 list_of_images=x_train,
    #                                                                 n=IMGS))
    #                       for i in range(NC)}
    #
    # for i in range(NC):
    #     images_for_classes[i] = torch.tensor(images_for_classes[i]).to(device)

    images_for_classes = mu.build_images_tensor(num_of_classes=NC,
                                                number_of_images=IMGS,
                                                list_of_labels=y_train,
                                                list_of_images=x_train).to(device)

    file_base_name = "first_MNIST_driver-with-an-almost-lemony-freshness"
    file_name_ext = ".tsv"
    t = int(time.time())
    file_name = file_base_name + '_' + str(t) + file_name_ext

    # need to build our neighbourhoods
    # work out our coordinates
    coords_3x3_list = ap.get_NxN_neighbourhood(1)
    coords_5x5_list = ap.get_NxN_neighbourhood(2)
    coords_7x7_list = ap.get_NxN_neighbourhood(3)
    coords_9x9_list = ap.get_NxN_neighbourhood(4)
    coords_11x11_list = ap.get_NxN_neighbourhood(5)

    # define the neighbourhoods
    coords_3x3_neighbours = [[ap.get_neighbourhood(hotspot=(i, j),
                                                   coords=coords_3x3_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]
    coords_5x5_neighbours = [[ap.get_neighbourhood(hotspot=(i, j),
                                                   coords=coords_5x5_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]
    coords_7x7_neighbours = [[ap.get_neighbourhood(hotspot=(i, j),
                                                   coords=coords_7x7_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]
    coords_9x9_neighbours = [[ap.get_neighbourhood(hotspot=(i, j),
                                                   coords=coords_9x9_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]
    coords_11x11_neighbours = [[ap.get_neighbourhood(hotspot=(i, j),
                                                   coords=coords_11x11_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]

    # stick 'em on the GPU
    neighbours_3x3 = torch.tensor(coords_3x3_neighbours, dtype=torch.int).to(device)
    neighbours_5x5 = torch.tensor(coords_5x5_neighbours, dtype=torch.int).to(device)
    neighbours_7x7 = torch.tensor(coords_7x7_neighbours, dtype=torch.int).to(device)
    neighbours_9x9 = torch.tensor(coords_9x9_neighbours, dtype=torch.int).to(device)
    neighbours_11x11 = torch.tensor(coords_11x11_neighbours, dtype=torch.int).to(device)

    neighbourhood_list = [neighbours_3x3, neighbours_5x5, neighbours_7x7, neighbours_9x9, neighbours_11x11]

    # create the repertoire
    repertoire = torch.empty(rep_shape, dtype=torch.uint8).to(device)
    for i in range(R_SZ):
        repertoire[i] = ap.generate_apc(shape=apc_shape)

    # yep, we *can* tidy up this code - but let's just get the damn thing working first!
    apc_distance = torch.empty(R_SZ, dtype=torch.float32).to(device)

    with open(file_name, mode='w') as out:
        with Pool(R_SZ) as pool:
            for it in range(ITERATIONS):
                # TODO: we need to make sure we get the distance for each image, then
                # TODO: we get the mean of each iteration
                for img_idx in range(IMGS):
                    args = string_build_arg_list(repertoire_=repertoire,
                                                    device_=device,
                                                    neighbourhood_list_=neighbourhood_list,
                                                    images_for_classes_=images_for_classes,
                                                    R_SZ_=R_SZ,
                                                    C_SZ_=C_SZ,
                                                    IMGS_=IMGS,
                                                    rho=0.99)
                    results = pool.starmap(ap.batch_process_clone_and_hypermutate, args)
                    for idx, (apc, aff) in enumerate(results):
                        apc_distance[idx] = aff
                        repertoire[idx] = apc
                    # we divide the affinity by the number of images
                    mean_affinity = float(apc_distance.sum() / R_SZ)
                    print(f"{it} : {mean_affinity}")
                    out.write(f"{str(it)}\t{str(mean_affinity)}\n")

        end = time.time()
        print(f"Execution time: {end - start}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
