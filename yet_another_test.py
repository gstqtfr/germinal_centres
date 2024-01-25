#!/usr/bin/env python

from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import torch_apc_utils1 as ap
import numpy as np
import torch
import time
from PIL import Image, ImageOps

def main():

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    ITERATIONS = 50

    # load up the image
    data_directory = "/home/johnny/Documents/datasets/cats-vs-dogs/PetImages"
    img_shape = (224, 224)
    img_host = Image.open(data_directory + '/cat/1.jpg')
    img_host = ImageOps.grayscale(img_host)
    img = img_host.resize(size=img_shape)
    img = torch.from_numpy(np.array(img)).to(device)

    apc_shape = (16, 16)
    R_SZ = 8
    C_SZ = 8
    rep_shape = (R_SZ, *apc_shape)

    file_base_name = "random_neighbour_selection_gc"
    file_name_ext = ".tsv"
    t = int(time.time())
    file_name = file_base_name + '_' + str(t) + file_name_ext

    # create the repertoire
    repertoire = torch.empty(rep_shape, dtype=torch.uint8).to(device)
    for i in range(R_SZ):
        repertoire[i] = ap.generate_apc(shape=apc_shape)

    # get our target size
    # calculate_target_size(img_size: int, paratope_size: int) -> int:
    sz_ = ap.calculate_target_size(img_size=img_shape[0], paratope_size=8)
    sz_squared = sz_ * sz_

    # work out our coordinates
    coords_3x3_list = ap.get_NxN_neighbourhood(1)
    coords_5x5_list = ap.get_NxN_neighbourhood(2)
    coords_7x7_list = ap.get_NxN_neighbourhood(3)

    # define the neighbourhoods
    coords_3x3_neighbours = [[ap.get_neighbourhood(hotspot=[i, j],
                                                   coords=coords_3x3_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]
    coords_5x5_neighbours = [[ap.get_neighbourhood(hotspot=[i, j],
                                                   coords=coords_5x5_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]
    coords_7x7_neighbours = [[ap.get_neighbourhood(hotspot=[i, j],
                                                   coords=coords_7x7_list,
                                                   max_=apc_shape[0]) for j in range(apc_shape[1])]
                             for i in range(apc_shape[0])]

    neighbours_3x3 = torch.tensor(coords_3x3_neighbours, dtype=torch.int).to(device)
    neighbours_5x5 = torch.tensor(coords_5x5_neighbours, dtype=torch.int).to(device)
    neighbours_7x7 = torch.tensor(coords_7x7_neighbours, dtype=torch.int).to(device)

    neighbourhood_list = [neighbours_3x3, neighbours_5x5, neighbours_7x7]

    print(f"Building the arguments")
    args = [(repertoire[idx], idx, device, neighbourhood_list, \
             img, C_SZ, sz_squared) for idx in range(R_SZ)]

    # yep, we *can* tidy up this code - but let's just get the damn thing working first!
    apc_affinity = torch.empty(R_SZ, dtype=torch.float32).to(device)

    with open(file_name, mode='w') as out:
        with Pool(R_SZ) as pool:
            for it in range(ITERATIONS):
                results = pool.starmap(ap.clone_and_hypermutate_with_neighbour_selection, args)
                # do we need this to(device) thing every time?
                for idx, (apc, aff) in enumerate(results):
                    apc_affinity[idx] = aff
                    repertoire[idx] = apc
                mean_affinity = float(apc_affinity.sum() / R_SZ)
                print(f"{it} : {mean_affinity}")
                out.write(f"{str(it)}\t{str(mean_affinity)}\n")

    end = time.time()
    print(f"Execution time: {end - start}")

    # TODO: this is consistently the worst-performing APC; find out why the hell this is
    #paratopes = ap.get_paratope_frequency(repertoire_=repertoire).to(device).reshape((16,16))
    #dead_parrot = torch.argmax(apc_affinity)
    #print(f"{it}: doing a splat job on APC {dead_parrot}, with affinity {apc_affinity[dead_parrot]}")
    #hotspot = torch.randint(0, size=(2,), high=16).to(device)
    #repertoire[dead_parrot] = ap.mat_wrap_copy(source=paratopes, dest=repertoire[dead_parrot], offset=hotspot)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()