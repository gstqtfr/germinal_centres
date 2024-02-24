import torch
from typing import Tuple, List, Union
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


def select_neighbourhood(neighbourhoods):
    """Given a list of neighbourhoods, randomly selects & returns one
    :param neighbourhoods: list of various-sized neighbourhoods
    :type neighbourhoods: list of torch tensors
    :return: randomly-generated offset into the list
    :rtype: uint
    """
    N = len(neighbourhoods)
    return int(torch.rand(1) * N)


def get_neighbourhood(hotspot: Tuple[int, int], coords: List[Tuple[int, int]], max_: int = 8) -> List[
    List[List[Tuple[int, int]]]]:
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


# def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
#     """
#     Find the Manhattan distance between two elements of a matrix. We're calculating the distance based on their
#     indices. For example if we're given the cell of a matrix at (0,0), & want to find the distance to (3,-2), we
#      have manhattan(0, 0, 3,-2) = 5.
#     :param x1: x-coordinate of first matrix cell
#     :type x1: int
#     :param y1: y-coordinate of first matrix cell
#     :type y1: int
#     :param x2: x-coord of second matrix cell
#     :type x2: int
#     :param y2: y-coord of second matrix cell
#     :type y2: int
#     :return: Manhattan distance between the two cells
#     :rtype: int
#     """
#     return int(math.fabs(x2 - x1) + math.fabs(y2 - y1))

def calculate_diffusion_coefficients_by_coordinates(hotspot, coord, rho, N=3):
    def manhattan_(x1: int, y1: int, x2: int, y2: int) -> int:
        return int(math.fabs(x2 - x1) + math.fabs(y2 - y1))

    distance = manhattan_(hotspot[0], hotspot[1], coord[0], coord[1])
    # we want an inverse relationship between the distance & the probability
    if distance > N:
        return 0.0
    elif distance == 0.0:
        return 1.0
    else:
        return 1.0 / (distance + 1.0)


def get_distance(img, apc):
    return LA.matrix_norm(apc.type(torch.float32) - img.type(torch.float32))


def scs(receptor_, img_, p=2.5, eps=1e-06):
    # we need to preserve the sign of the receptor
    signs = torch.sign(receptor_)
    cos = torch.nn.CosineSimilarity(eps=eps)
    similarity = cos(torch.abs(receptor_).type(torch.float), img_.type(torch.float))
    return signs * torch.pow(similarity, p)

# JKK: inspired by https://github.com/detkov/Convolution-From-Scratch

def add_padding(matrix: torch.Tensor,
                padding: Tuple[int, int]) -> torch.Tensor:
    """Adds padding to the matrix.

    Args:
        matrix (torch.tensor): Matrix, in the form of a Torch tensor, that needs to be padded.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        torch.tensor: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = torch.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix

    return padded_matrix

def check_params(matrix, kernel, stride, dilation, padding):
    params_are_correct = (isinstance(stride[0], int) and isinstance(stride[1], int) and
                          isinstance(dilation[0], int) and isinstance(dilation[1], int) and
                          isinstance(padding[0], int) and isinstance(padding[1], int) and
                          stride[0] >= 1 and stride[1] >= 1 and
                          dilation[0] >= 1 and dilation[1] >= 1 and
                          padding[0] >= 0 and padding[1] >= 0)
    assert params_are_correct, 'Parameters should be integers equal or greater than default values.'
    if not isinstance(matrix, torch.Tensor):
        # JKK: no idea if this will work - let's find out ...
        matrix = torch.tensor(matrix)
    n, m = matrix.shape
    matrix = matrix if list(padding) == [0, 0] else add_padding(matrix, padding)
    n_p, m_p = matrix.shape

    if not isinstance(kernel, torch.Tensor):
        kernel = torch.tensor(kernel)
    k = kernel.shape

    kernel_is_correct = k[0] % 2 == 1 and k[1] % 2 == 1
    assert kernel_is_correct, 'Kernel shape should be odd.'
    matrix_to_kernel_is_correct = n_p >= k[0] and m_p >= k[1]
    assert matrix_to_kernel_is_correct, 'Kernel can\'t be bigger than matrix in terms of shape.'

    h_out = torch.floor((n + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = torch.floor((m + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    out_dimensions_are_correct = h_out > 0 and w_out > 0
    assert out_dimensions_are_correct, 'Can\'t apply input parameters, one of resulting output dimension is non-positive.'

    return matrix, kernel, k, h_out, w_out


def conv2d(matrix: Union[List[List[float]], torch.Tensor],
           kernel: Union[List[List[float]], torch.Tensor],
           stride: Tuple[int, int] = (1, 1),
           dilation: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0)) -> torch.tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (Union[List[List[float]], torch.tensor]): 2D matrix to be convolved.
        kernel (Union[List[List[float]], torch.tensor]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        torch.tensor: 2D Feature map, i.e. matrix after convolution.
    """
    matrix, kernel, k, h_out, w_out = check_params(matrix, kernel, stride, dilation, padding)
    matrix_out = torch.zeros((h_out, w_out))

    b = k[0] // 2, k[1] // 2
    center_x_0 = b[0] * dilation[0]
    center_y_0 = b[1] * dilation[1]
    for i in range(h_out):
        center_x = center_x_0 + i * stride[0]
        indices_x = [center_x + l * dilation[0] for l in range(-b[0], b[0] + 1)]
        for j in range(w_out):
            center_y = center_y_0 + j * stride[1]
            indices_y = [center_y + l * dilation[1] for l in range(-b[1], b[1] + 1)]

            submatrix = matrix[indices_x, :][:, indices_y]

            matrix_out[i][j] = torch.sum(torch.multiply(submatrix, kernel))
    return matrix_out



def somatic_hypermutation_with_diffusion(clone, device, all_neighbours, hotspot, rho,
                                         low=0, high=256, dtype=torch.uint8):
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
                clone[x][y] = torch.randint(low=low, high=high, size=(1,), dtype=dtype)
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


def clone_and_hypermutate_with_neighbour_selection(apc_, idx_, image_idx_, device_,
                                                   neighbourhoods_, img_, C_SZ_,
                                                   rho=0.99, low=0, high=256, dtype=torch.uint8):

    # our base distance - if we get a clone that's better than this distance, we overwrite
    distance_ = get_distance(img_, apc_).to(device_)
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
                                                              rho=rho,
                                                              low=low,
                                                              high=high,
                                                              dtype=dtype)

        clone_distance = get_distance(img_, clone).to(device_)

        if clone_distance <= distance_:
            apc_ = clone
            distance_ = clone_distance

    # we have to clone the tensors before returning them
    apc_clone = apc_.clone()
    distance_clone = distance_.clone()

    return apc_clone, distance_clone


def batch_process_clone_and_hypermutate(apc_, idx_, device_, neighbourhoods_, images_, C_SZ_,
                                        rho=0.99, low=0, high=256, dtype=torch.uint8):
    """We call the clone-&-hypermutate code, & collect the results. We replace at the end of the batch
     of data."""

    # TODO: JKK: get rid when we're sorted
    print(f"batch_process_clone_and_hypermutate: low: {low}")
    print(f"batch_process_clone_and_hypermutate: dtype: {dtype}")

    # collect our (maybe) clones
    # this needs to be a tensor ...
    might_be_a_clone = []
    might_be_a_clone_affinity = torch.zeros(images_.shape[0])

    for image_idx, image in enumerate(images_):
        maybe_clone, maybe_clone_affinity = clone_and_hypermutate_with_neighbour_selection(
            apc_=apc_,
            idx_=idx_,
            image_idx_=image_idx,
            device_=device_,
            neighbourhoods_=neighbourhoods_,
            img_=image,
            C_SZ_=C_SZ_,
            rho=rho,
            low=low,
            high=high,
            dtype=dtype)

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
