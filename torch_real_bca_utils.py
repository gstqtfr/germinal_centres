import torch
import numpy as np
import struct

def clone(antibodies, number_of_clones, device):
    clones = torch.zeros((antibodies.shape[0], number_of_clones), dtype=torch.float64).to(device)
    for idx in torch.arange(antibodies.shape[0]):
        clones[idx] = torch.tile(antibodies[idx], (number_of_clones,))
    return clones

# let's split this up a little bit - the code's a little "side-effecty"
def get_mutations(M, N, epsilon, device):
    return torch.empty(size=(M, N,)).normal_(mean=0.,std=epsilon).to(device)

def mutate(clones, eps, device):
    mutations = get_mutations(M=clones.shape[0], N=clones.shape[1], epsilon=eps, device=device)
    return clones + mutations

def objective(x):
    """Test function.

    Functions are in torch, so they require x to be a torch tensor.

    Here's a useful range to define the antibodies in:

    # define range for input
    r_min, r_max = -10.0, 10.0

    The optimal x-value is approx. 0.67956:

    # optimal x-value to plug into the objective function
    x_optima = torch.tensor([0.67956]).to(device)

    which evaluates to approx. -1.6768644187726358
    """
    return -(x + torch.sin(x) * torch.exp(torch.pow(-x, 2.)))

def objective1(x):
    """Objective function to test the real-valued BCA.

        Here's a useful range to define the antibodies in:

    # define range for input
    r_min, r_max = -2.7, 7.5

    The optimal x-value is approx. 5.145735:

    # optimal x-value to plug into the objective function
    x_optima = torch.tensor([5.145735]).to(device)

    which evaluates to approx. -1.8995993137

    """
    return torch.sin(x) + torch.sin((10.0 / 3.0) * x)

def distance(antibodies, optima, f):
    return torch.pow(torch.pow((f(antibodies) - optima), 2.), 0.5)

def get_highest_affinity_index(affinity):
    result = torch.min(affinity, dim=0, keepdim=False)
    return result.indices

def get_highest_affinity_clone(antibodies, affinities):
    return antibodies[get_highest_affinity_index(affinities)]


def apply_real_valued_BCA(x_optima, func, r_min, r_max, trial, epsilon=2.5, tolerance=1e-6, iterations=50000, N=50,
                          C=50, halving=100):
    # initialise the population
    antibodies = torch.distributions.uniform.Uniform(r_min, r_max).sample([P]).to(device)

    # get out x-value on the correct device
    optima = torch.tensor([x_optima]).to(device)
    # then get our y-value
    optimal_value = torch.tensor([func(optima)]).to(device)

    # set up oour termination condition
    terminate = False
    optimal_antibody = 0.
    optimal_affinity = 0.

    for i in range(iterations):
        if terminate:
            print(f"trial {trial} iteration {i} antibody {optimal_antibody} is within {tolerance} to {x_optima}")
            print(f"trial {trial} iteration {i} antibody affinity: {optimal_affinity}")
            print(f"trial {trial} optimal antibody evaluates to {func(optimal_antibody)}")
            print(f"trial {trial} terminating at iteration {i}")
            return optimal_antibody, optimal_affinity, i

        if i % halving == 0:
            epsilon = epsilon / 2.

        # get the affinity
        affinities = distance(antibodies=antibodies, optima=optimal_value, f=func)
        # we'd better see whether or not the clones are heading off to infinity or not
        clones = mutate(clone(antibodies=antibodies, number_of_clones=C, device=device), eps=epsilon, device=device)
        # get the affinities of the clones
        clone_affinities = distance(antibodies=clones, optima=optimal_value, f=func)

        # iterative, but it kind of needs to be ...
        for idx in range(antibodies.shape[0]):
            clone_idx = get_highest_affinity_index(clone_affinities[idx])
            clone_affinity = clone_affinities[idx][clone_idx]
            if clone_affinity < tolerance:
                terminate = True
                optimal_antibody = clones[idx][clone_idx]
                optimal_affintiy = clone_affinity
                break
            elif clone_affinity < affinities[idx]:
                antibodies[idx] = clones[idx][clone_idx]
                # if we're replacing, we have already got the highest affinity clone for this antibody; so
                # we iterrupt the loop & go to the next antibody
                continue

    return None, None, i
