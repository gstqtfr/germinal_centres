import torch
from typing import Tuple, List
from torch import linalg as LA
import math


def generate_apc(shape: Tuple[int, int], low=0, high=256, dtype=torch.uint8) -> torch.tensor:
    """
    Generates a 2D matrix representing the receptor of an antigen presenting cell (APC).
    The matrix is of shape dimensions, and of type unsigned int. It's randomly generated
    with each element in the range [0,256).
    :param shape: shape of 2D matrix required, e.g. [8,8], [16,16], et.
    :type shape: tuple of unsigned ints
    :param low: lowest end of the range of the randomly generated matrix, usually 0.
    :type low: unsigned int
    :param high: high end of the range of the matrix, 256 as default, since we're using unsigned ints.
    :type high: int, in this case - the random operation interval is exclusive at the upper end.
    :param dtype: required type - default is unsigned int
    :type dtype: a torch dtype specification
    :return: a randomly-initialised 2D matrix
    :rtype: torch tensor of the given dimension and type
    """
    return torch.randint(size=shape, low=low, high=high, dtype=dtype)


def get_NxN_neighbourhood(N: int) -> List[Tuple[int, int]]:
    """Creates a square, 2D neighbourhood of a given dimension, surrounding a central hotspot.
    N=1 generates a 3x3 neighbourhood, N=2 a 5x5 neighbourhood, etc.
    For N=1, we generate matrix offsets centred on (0,0), like the following:

    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    These offsets allows us to easily reference the matrix at the given indices.
    :param N: the size of the 2D neighbourhood surrounding a central square.
    :type N: int
    :return: the matrix offsets which surround (& include) the central square, or hotspot.
    :rtype: List of indicies (coordinates) of neighbouring squares in 2-tuple format.
    """
    return [(i, j) for i in range(-N, N + 1) for j in range(-N, N + 1)]


def get_paratope_frequency(repertoire_, max_=256):
    """Gets the frequency of 1-paratopes expressed as a frequency table in the repertoire.
    :param repertoire_: repertoire of APC receptors - 2D matrices in torch tensor format, so a 3D tensor
    :type repertoire_: 3D torch tensor of unsigned integers
    :param max_: high (excluded) end of the [0,256) interval
    :type max_: int
    :return: a frequency table of the paratopes expressed in the repertorie.
    :rtype: 2D matrix
    """
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
    """Given a list of neighbourhoods, randomly selects & returns one
    :param neighbourhoods: list of various-sized neighbourhoods
    :type neighbourhoods: list of torch tensors
    :return: randomly-generated offset into the list
    :rtype: uint
    """
    N = len(neighbourhoods)
    return int(torch.rand(1) * N)


def get_neighbourhood(hotspot: Tuple[int, int], coords: List[Tuple[int, int]], max_: int = 8) -> List[List[List[Tuple[int, int]]]]:
    """
    Generates a neighbourhood for each of the possible neighbourhoods in a given matrix.
    We take a hotspot - a coordinate in the matrix, say (x,y) - & we produce any neighbouring squares in the
    matrix, using coords as our offsets. This is not toroidal - any squares not in the immediate vicinity are
    given as (-1,-1), which means unreachable; we don't wrap around to the other side of the matrix, since this
    element is by definition non-local.

    We return a list of (N,N, hood=neighbourhood-size, 2), where (N,N) is the size of the matrix, hood is the
    size of the neighbourhood (e.g. 9, 25, 36, ...), and 2 is the tuple giving the coordinates.

    As an example, say we have an 3x3 matrix as defined by coords, & we want to generate the neighbourhoods for [0,0]. Given this
    coordinate, quite a few of the entries are unreachable - they'd require us to wrap aound to the other side
    of the matrix, or the bottom of the matrix. But we are only interested in contiguous squares in the matrix.
    So we'd produce the following:

    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [0, 0],
    [0, 1],
    [-1, -1],
    [1, 0],
    [1, 1]

    However, a hotspot located more centrally in the matrix, such as [2,2], would produce:

    [[1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3]]

    :param hotspot: coordinate we want to find the neighbourhood of
    :type hotspot: tuple[int, int]
    :param coords: list of tuples (coordinates) centred on the central square, for example as created by get_NxN_neighbourhood
    :type coords: list[tuple[int, int]]
    :param max_: maximum size of the side of the APC receptor matrix, e.g. 8 for an 8x8 matrix
    :type max_:int
    :return: a list of lists of all the neighbours for the hotspot, with unreachable coordinates given a (-1,-1)
    :rtype: List[List[List[Tuple[int, int]]]]
    """
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


def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Find the Manhattan distance between two elements of a matrix. We're calculating the distance based on their
    indices. For example if we're given the cell of a matrix at (0,0), & want to find the distance to (3,-2), we
     have manhattan(0, 0, 3,-2) = 5.
    :param x1: x-coordinate of first matrix cell
    :type x1: int
    :param y1: y-coordinate of first matrix cell
    :type y1: int
    :param x2: x-coord of second matrix cell
    :type x2: int
    :param y2: y-coord of second matrix cell
    :type y2: int
    :return: Manhattan distance between the two cells
    :rtype: int
    """
    return int(math.fabs(x2 - x1) + math.fabs(y2 - y1))


def calculate_diffusion_coefficients_by_coordinates(hotspot, coord, rho, N=3):
    distance = manhattan(hotspot[0], hotspot[1], coord[0], coord[1])
    # we want an inverse relationship between the distance & the probability
    if distance > N:
        return 0.0
    elif distance == 0.0:
        return 1.0
    else:
        return 1.0 / (distance + 1.0)


def somatic_hypermutation_with_diffusion(clone, device, all_neighbours, hotspot, rho, max_x=256):
    """Get the neighbourhood for this hotspot, based on the neighbours parameter."""

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

def get_distance(img, apc):
    return LA.matrix_norm(apc.type(torch.float32) - img.type(torch.float32))

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


def batch_process_clone_and_hypermutate(apc_, idx_, device_, neighbourhoods_, images_, C_SZ_, rho=0.99):
    """We call the clone-&-hypermutate code, & collect the results. We replace at the end of the batch
     of data."""

    # collect our (maybe) clones
    # this needs to be a tensor ...
    might_be_a_clone = []
    might_be_a_clone_affinity = torch.zeros(images_.shape[0])

    for image_idx, image in enumerate(images_):
        maybe_clone, maybe_clone_affinity = clone_and_hypermutate_with_neighbour_selection(apc_=apc_,
                                                                                           idx_=idx_,
                                                                                           image_idx_=image_idx,
                                                                                           device_=device_,
                                                                                           neighbourhoods_=neighbourhoods_,
                                                                                           img_=image,
                                                                                           C_SZ_=C_SZ_,
                                                                                           rho=rho)

        might_be_a_clone.append(maybe_clone)
        might_be_a_clone_affinity[image_idx] = maybe_clone_affinity

    # now we take the average of these values, & assign them to the APC &
    # it's corresponding affinity
    # of course, we could have all sorts of different operations here - doesn't have to be
    # the mean value of all these. note also that, if we don't have a higher-affinity clone
    # then we return the original APC from the clone-&-hypermutate code, so we just write
    # back what we've already got in the repertoire

    # cat to the right type (we may overflow 256, since these are unsigned 8-bit ints)
    might_be_a_clone = [e.to(torch.int32) for e in might_be_a_clone]
    # stack 'em & sum 'em
    maybe_maybe_maybe = torch.stack(might_be_a_clone, dim=0).sum(dim=0)
    # get the mean of the values, & cast back to unsigned ints
    maybe_maybe_maybe = (maybe_maybe_maybe / len(might_be_a_clone)).to(torch.uint8)

    # get the mean affinity, & replace
    # we get the affinity of each "clone", divided by the number of images
    mean_affinity = might_be_a_clone_affinity.sum() / might_be_a_clone_affinity.shape[0]

    # we automatically replace the APC receptor
    apc_ = maybe_maybe_maybe.clone()

    return apc_, mean_affinity




def clone_and_hypermutate_with_neighbour_selection(apc_, idx_, image_idx_, device_, neighbourhoods_,
                                                   img_, C_SZ_, rho=0.99):
    # TODO: JKK: yeah, we don;t need to normlaise the distance here because they're the same size
    #distance_ = get_normalised_distance(img_, apc_).to(device_)

    distance_ = get_distance(img_,apc_).to(device_)
    #affinity_ = get_affinity(distance_).to(device_)

    # create our clonal pool
    clones = torch.tile(apc_, (C_SZ_, 1, 1)).to(device_)

    # hypermutate our clones
    for c_id, clone in enumerate(clones):
        neighbourhood_idx = select_neighbourhood(neighbourhoods_)
        neighbourhood_ = neighbourhoods_[neighbourhood_idx]
        hotspot = torch.randint(0, size=(2,), high=clone.shape[0]).to(device_)
        # JKK: this WILL THROW AN ERROR in the module, so remember to remove ap. ...
        clone, mutants = somatic_hypermutation_with_diffusion(clone=clone,
                                                              device=device_,
                                                              all_neighbours=neighbourhood_,
                                                              hotspot=hotspot,
                                                              rho=rho)
        #clone_distance = get_normalised_distance(img_, clone).to(device_)
        clone_distance = get_distance(img_, clone).to(device_)
        #clone_affinity = get_affinity(clone_distance).clone().to(device_)

        if clone_distance <= distance_:
            print(f"image {idx_} apc {c_id} affinity {distance_} adding with distance {clone_distance}")
            apc_ = clone
            distance_ = clone_distance

    # we have to clone the tensors before returning them
    apc_clone = apc_.clone()
    distance_clone = distance_.clone()

    return apc_clone, distance_clone


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
