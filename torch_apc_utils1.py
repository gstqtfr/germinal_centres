import torch
from typing import Tuple
from torch import linalg as LA
import math


def generate_apc(shape: Tuple[int, int], low=0, high=256, dtype=torch.uint8):
    return torch.randint(size=shape, low=low, high=high, dtype=dtype)


def get_NxN_neighbourhood(N):
    return [(i, j) for i in range(-N, N + 1) for j in range(-N, N + 1)]


def get_paratope_frequency(repertoire_, max_=256):
    """Gets the frequency of 1-paratopes expressed as a frequency table in the repertoire."""
    # we need the repertoire size * the size of the matrix
    # that is, repertoire size * matrix rows * matrix cols
    bincount_sz = repertoire_.shape[0] * repertoire_.shape[1]
    p_mask_ = torch.zeros((bincount_sz, max_), dtype=torch.int32)
    idx = 0
    for i in range(repertoire_.shape[0]):
        for j in range(repertoire_.shape[1]):
            p_mask_[idx] = torch.bincount(repertoire_[i][j], minlength=max_)
            idx = idx + 1
    return torch.sum(p_mask_, dim=0)


def get_paratope_locations(repertoire_, device_, max_=256):
    loci_ = [[] for _ in range(max_)]

    for idx in range(repertoire_.shape[0]):
        for i in range(repertoire_.shape[1]):
            for j in range(repertoire_.shape[2]):
                p = repertoire_[idx, i, j]
                locus = (i, j)
                loci_[p].append(locus)

    loci_shape = max_

    loci = torch.zeros(loci_shape, dtype=torch.int32).to(device_)

    for idx in range(max_):
        loci[idx] = len(loci_[idx])

    return loci


def initialise_icol(N=10, image_shape=(8, 8)):
    icol_dims = (image_shape[0], image_shape[1], N)
    icol = torch.full(size=icol_dims, fill_value=-1)
    return icol


def select_neighbourhood(neighbourhoods):
    """Given a list of neighbourhoods, randomly selects & returns one"""
    N = len(neighbourhoods)
    return int(torch.rand(1) * N)


def get_neighbourhood(hotspot, coords, max_=8):
    def check_range(idx, min_=0, max=max_):
        return min_ <= idx < max

    x = hotspot[0]
    y = hotspot[1]

    # x,y is the [0,0] coord - the centre of the coords

    neighbours = []

    for cursor in coords:
        offset_x = cursor[0]
        offset_y = cursor[1]
        new_x = x + offset_x
        new_y = y + offset_y
        if check_range(new_x) and check_range(new_y):
            neighbours.append([new_x, new_y])
        else:
            neighbours.append([-1, -1])

    return neighbours


def manhattan(x1, y1, x2, y2):
    return int(math.fabs(x2 - x1) + math.fabs(y2 - y1))


def calculate_diffusion_coefficients_by_coordinates(hotspot, coord, rho, N=3):
    distance = manhattan(hotspot[0], hotspot[1], coord[0], coord[1])
    # print(f"distance between ({hotspot[0]},{hotspot[1]}) and ({coord[0]},{coord[1]}) is {distance}")
    # we want an inverse relationship between the distance & the probability
    if distance > N:
        return 0.0
    elif distance == 0.0:
        return 1.0
    else:
        return 1.0 / (distance + 1.0)


def somatic_hypermutation_with_diffusion(clone, device, all_neighbours, hotspot, rho, max_x=256):
    """Get the neighbourhood for this hotspot, based on the neighbours parameter."""

    # print(f"hotspot info: {get_DS_info(hotspot)}")
    # print(f"clone info: {get_DS_info(clone)}")
    # print(f"device: {device}")
    # print(f"all_neighbours info: {get_DS_info(all_neighbours)}")
    # print(f"rho: {rho}")

    neighbourhood = all_neighbours[hotspot[0], hotspot[1]]

    # we use l to store the hotspots that are actually mutated
    # we have to use a list here because we don't know how large it's going to be
    l = []

    for site in neighbourhood:
        x = site[0]
        y = site[1]
        if x != -1 and y != -1:
            diffusion_coefficient = calculate_diffusion_coefficients_by_coordinates(hotspot=hotspot, coord=site,
                                                                                    rho=rho)
            if torch.rand(1) < diffusion_coefficient:
                clone[x][y] = torch.randint(low=0, high=max_x, size=(1,), dtype=torch.uint8)
                # keep track of the hostpots & mutations so we know which ones are successfully adaptations
                l.append((x, y, clone[x][y]))

    # now we create a tensor to hold the info
    mutant_dims = (len(l), 3)
    # we call to(device) here since we do't know how big the mutants array is going to be
    mutants = torch.zeros(size=mutant_dims, dtype=torch.uint8).to(device)

    for idx, m in enumerate(l):
        mutants[idx][0] = m[0]
        mutants[idx][1] = m[1]
        mutants[idx][2] = m[2]

    return clone, mutants


def get_image_patch(img, x, y, width, height):
    return img[x:x + width, y:y + height]


def get_normalised_distance(img, apc):
    """Performs a 'sliding-window' distance metric over the image."""
    width = apc.shape[0]
    height = apc.shape[1]

    # we need to calculate the number of matrix norms we're going
    # to return
    L = []

    for x in range(img.shape[0] - width + 1):
        for y in range(img.shape[1] - height + 1):
            patch = get_image_patch(img, x, y, width, height)
            # we now get the norm of the difference between apc & the patch
            thing = LA.matrix_norm(apc.type(torch.float32) - patch.type(torch.float32))
            L.append(thing)

    return torch.Tensor(L)


def get_affinity(distances):
    """Gets the mean distance."""
    return distances.sum() / distances.shape[0]


def clone_and_hypermutate_with_neighbour_selection(apc_, idx_, device_, neighbourhoods_, img_, C_SZ_, rho=0.99):
    distance_ = get_normalised_distance(img_, apc_).to(device_)
    affinity_ = get_affinity(distance_).to(device_)

    # create our clonal pool
    clones = torch.tile(apc_, (C_SZ_, 1, 1)).to(device_)

    neighbourhood_idx = select_neighbourhood(neighbourhoods_)
    neighbourhood_ = neighbourhoods_[neighbourhood_idx]

    # hypermutate our clones
    for c_id, clone in enumerate(clones):
        hotspot = torch.randint(0, size=(2,), high=clone.shape[0]).to(device_)
        # JKK: this WILL THROW AN ERROR in the module, so remember to remove ap. ...
        clone, mutants = somatic_hypermutation_with_diffusion(clone=clone,
                                                              device=device_,
                                                              all_neighbours=neighbourhood_,
                                                              hotspot=hotspot,
                                                              rho=rho)
        clone_distance = get_normalised_distance(img_, clone).to(device_)
        clone_affinity = get_affinity(clone_distance).clone().to(device_)

        if clone_affinity <= affinity_:
            print(f"apc {idx_} affinity {affinity_} replacing with clone with affinity {clone_affinity}")
            apc_ = clone
            affinity_ = clone_affinity

    # we have to clone the tensors before returning them
    apc_clone = apc_.clone()
    affinity_clone = affinity_.clone()

    return apc_clone, affinity_clone


def mat_wrap_copy(source, dest, offset):
    """Copies from source to dest with overlap (toroidal) starting at offset."""
    sx, sy = source.shape
    dx, dy = dest.shape
    x, y = offset

    for i in range(dx):
        ix = (x + i) % sx
        for j in range(dy):
            jy = (y + j) % sy
            dest[i, j] = source[ix, jy]

    return dest


def calculate_target_size(img_size: int, paratope_size: int) -> int:
    num_pixels = 0

    # From 0 up to img size (if img size = 224, then up to 223)
    for i in range(img_size):
        # Add the kernel size to the current i
        added = i + paratope_size
        # It must be lower than the image size
        if added <= img_size:
            # Increment if so
            num_pixels += 1

    return num_pixels
