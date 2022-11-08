import os
import random
import numpy as np
import pickle as pkl
import numpy as np

from typing import Tuple, Union
from numpy.random import default_rng
from numpy.linalg import norm


"""

Sampling functions for objective

"""

def _stationary_sample(P, n_samples=1):

    data_size, _ = P.shape
    index = random.sample(range(data_size), n_samples)
    batch_sample =  P[index]
    batch_mean = np.mean(batch_sample, axis=0)
    return np.split(batch_mean, 2, axis=-1)


def _demand_sample(x,y,a,b,A,B,C,D):

    a += A*x + B*y
    b += C*x + D*y
    
    return a, b


def max_power(file):
    
    file_name = file.split(".")[0]
    
    return int(file_name.split("_")[-1])

def get_charging_rate(capacity):

    if capacity <= 150:
        rate = 1.0

    else:
        rate = 4.0 

    return rate

"""

Sampling constructor and helpers

"""

def build_sampler(
    directory,
    time,
    process: bool = True,
    return_mean: bool = False
    ):

    # get file names and randomly shuffle
    file_names = os.listdir(directory)
    random.shuffle(file_names)

    # get station capacity from file names
    station_capacities = np.array([ max_power(file) for file in file_names])
    f,g = np.split(
        np.array(
            [ get_charging_rate(capacity) 
              for capacity in station_capacities
            ]
        ), 
        2
    )

    # load data and pick a time
    data = np.stack([
        pkl.load(open(os.path.join(directory,file),"rb")) 
        for file in file_names
    ],
    axis=2
    )
    
    distribution = data[time]
    mean = np.mean(data,axis=0)
    station_wise_variance = np.sqrt(np.mean(np.multiply(data,data),axis=0))

    if process:

        distribution = (distribution - mean) / station_wise_variance
    
    elif not process:
        pass

    if return_mean:

        return distribution, f, g, mean 

    elif not return_mean:

        return distribution, f, g

# def get_mean(
#     data: np.ndarray, 
#     split: bool = True
#     ):
    
#     mean = np.mean(data,axis=0)
    
#     if split: 
#         return np.split(mean,2)
    
#     elif not split:
#         return mean
    
#     else: 
#         raise NotImplementedError(
#             f"Split flag {split} not allowed."
#         )


"""

Random vector generators

"""
    
def sphere_sample(n, rng = default_rng()):
    
    u = rng.standard_normal(size=(n,))

    return u/norm(u)

def box_sample(a,b,size, rng = default_rng()):

    return rng.uniform(a,b,size)