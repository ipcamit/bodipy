"""
This program is part of MOLDIS: The bigdata analytics platform. Accompanying manuscript
and the complementary web interface can be accessed at : https://moldis.tifrh.res.in/data/bodipy
Python Requirements:
numpy, scipy, scikit-learn, QML
System Requirements:
MOPAC, Obabel, write permission in running directory
This file is specifically for Genetic Optimization problem. Although structure is same,
not all keywords carry same meaning or uitility. 
Licence: MIT
"""
# ========IMPORTS================
import numpy as np
from sklearn import gaussian_process
import scipy as sp
import argparse as ap
from GenerateBodipy import GenerateBodipy
from GenerateSLATM import GenerateSLATM
from GeneticOptimization import GA

# ========ARGPARSER=================
parser = ap.ArgumentParser(description="This program takes in a target value in eV and yield BODIPY molecules\
    closer to that value. Only improvement over previous evaluations are displayed. ")
parser.add_argument("target", type=float, help="Target S0->S1 value, in eV")
parser.add_argument("--data", "-d", type=str, default="./data",
                    help="Location of datafiles, default = ./data")
parser.add_argument("--seed", "-s", type=int, default=20,
                    help="Number of initial evaluations (Parent population) to build the GA model, default = 20")
parser.add_argument("--iter", "-i", type=int, default=200,
                    help="Maximum number of iterations (generations), default = 200")
parser.add_argument("--mut", "-m", type=float, default=0.01,
                    help="Probability of mutation of each group, default = 0.01")
parser.add_argument("--tol", "-t", type=float, default=0.001,
                    help="Tolerance, stop iterations once absolute error is less then, default = 0.001")

args = parser.parse_args()

# ==============================================
n_iter = args.iter
n_seeds = args.seed
target = args.target
data_dir = args.data
tol = args.tol
mut = args.mut
# ================================================
print("Searching for 7D BODIPY near {:f} eV".format(target))
print("Reading ML model from {}".format(data_dir))
print("Generations {:d}; Parent population {:d}".format(n_iter, n_seeds))
print("Starting Genetic optization")
# ================================================


# ===================CLASSES==================================
class KRRModel:
    """
    This class contains the KRR ML machine. The coefficients $\\alpha$
    and descriptor, $d$, will be loaded from location <data>, using files 
    named desc.npy and coeff.npy. Hyperaparameter$\\sigma$ is defined on 
    the basis of median search. Energy is evaluated as
    $$
    E = \\sum_i \\alpha_i * exp(-\\frac{\\sum_j |(d_i - d_j)|}{\\sigma}).
    $$
    """

    def __init__(self, target, data_dir=data_dir):
        self.desc = np.load("{}/desc.npy".format(data_dir))
        # self.desc = self.desc[0:2000,:]
        # self.desc = self.desc.astype("float")
        self.coeff = np.load("{}/coeff.npy".format(data_dir))
        # self.coeff = self.coeff[0:2000]
        # self.sigma = 26.57
        self.sigma = 840.087535153
        self.target = target
        self.bodipy_generator = GenerateBodipy()
        self.slatm_generator = GenerateSLATM()

    def get_s0s1(self, descriptor):
        desc_diff = np.exp(-np.sum(np.abs(self.desc -
                           descriptor), axis=1)/self.sigma)
        s0s1 = np.sum(desc_diff * self.coeff)
        return s0s1

    def get_loss(self, sub_array: np.ndarray):
        """
        Get loss function.
        loss function = -(E - Target)**2; 
        for inverted parabola to be optimzed using EI
        param: descriptor (1x322 numpy array)
        return: scalar loss
        """
        descriptor = self.gen_descriptor(sub_array)
        s0s1 = self.get_s0s1(descriptor)
        return (s0s1 - self.target)**2

    def gen_descriptor(self, sub_array:np.ndarray):
        """
        Generate SLATM descriptor based on input array.
        Input array format: for array of len L, L/2 = n_groups 
        [ <L/2 positions>, <L/2 substitution> ]
        param: Len 1x(2*n_groups) array
        return: 1x18023 slatm descriptor 
        """
        # print(sub_array)
        positions = sub_array[0:len(sub_array)//2].astype(int)
        substitutions= sub_array[len(sub_array)//2:len(sub_array)].astype(int)
        descriptor = np.zeros((1, 18023))
        self.bodipy_generator(list(positions.flatten()), list(substitutions.flatten()))
        descriptor = self.slatm_generator()
        return descriptor.reshape(1, -1)



# =======================INIT======================
kkr = KRRModel(target=target, data_dir=data_dir)
ga = GA(obj_func=kkr.get_loss, 
        target=target, 
        max_generation=n_iter, 
        mutation_rate=mut, 
        population_size=n_seeds)
ga.optimize(tol)

