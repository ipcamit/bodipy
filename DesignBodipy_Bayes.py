"""
This program is part of MOLDIS: The bigdata analytics platform. Accompanying manuscript
and the complementary web interface can be accessed at : https://moldis.tifrh.res.in/data/bodipy
Python Requirements:
numpy, scipy, scikit-learn, QML
System Requirements:
MOPAC, Obabel, write permission in running directory
Licence: MIT
"""
# ========IMPORTS================
import numpy as np
from sklearn import gaussian_process
import scipy as sp
import argparse as ap
from GenerateBodipy import GenerateBodipy
from GenerateSLATM import GenerateSLATM

# ========ARGPARSER=================
parser = ap.ArgumentParser(description="This program takes in a target value in eV and yield BODIPY molecules\
    closer to that value. Only improvement over previous evaluations are displayed. ")
parser.add_argument("target", type=float, help="Target S0->S1 value, in eV")
parser.add_argument("--group", "-g", type=int, default=2,
                    help="# of substitutions, default = 2, range 2-7")
parser.add_argument("--data", "-d", type=str, default="./data",
                    help="Location of datafiles, default = ./data")
parser.add_argument("--restart", "-r", type=int, default=5,
                    help="# of evaluations for single EI evaluation.\
                                More evaluations give more robust minima, with higher computation cost, default = 5")
parser.add_argument("--exploration", "-x", type=float, default="0.01",
                    help="Exploitation vs Exploration parameter, default = 0.01, range 0 100")
parser.add_argument("--seed", "-s", type=int, default=5,
                    help="Number of initial evaluations to build the GP model, default = 5")
parser.add_argument("--iter", "-i", type=int, default=200,
                    help="Maximum number of iterations. Largest covariance matrix in GP = iter + seed, default = 200")
parser.add_argument("--tol", "-t", type=float, default=0.001,
                    help="Tolerance, stop iterations once absolute error is less then, default = 0.001")


args = parser.parse_args()

# ==============================================
n_groups = args.group
n_iter = args.iter
n_seeds = args.seed
ex_v_ex = args.exploration
n_restart = args.restart
target = args.target
data_dir = args.data
tol = args.tol
# ================================================
print("Searching for {:d}D BODIPY near {:f} eV".format(n_groups,  target))
print("Reading ML model from {}".format(data_dir))
print("Iterations {:d}; Initial evaluations {:d}".format(n_iter, n_seeds))
print("Bayesian opt. parameters:\n Exploration/Exploitation \
param: {:f}; Eval. per EI: {:d}".format(ex_v_ex, n_restart))
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

    def get_loss(self, descriptor):
        """
        Get loss function.
        loss function = -(E - Target)**2; 
        for inverted parabola to be optimzed using EI
        param: descriptor (1x322 numpy array)
        return: scalar loss
        """
        s0s1 = self.get_s0s1(descriptor)
        return -(s0s1 - self.target)**2

    def gen_descriptor(self, sub_array:np.ndarray):
        """
        Generate SLATM descriptor based on input array.
        Input array format: for array of len L, L/2 = n_groups 
        [ <L/2 positions>, <L/2 substitution> ]
        param: Len 1x(2*n_groups) array
        return: 1x18023 slatm descriptor 
        """
        positions = sub_array[0:len(sub_array)//2].astype(int)
        substitutions= sub_array[len(sub_array)//2:len(sub_array)].astype(int)
        descriptor = np.zeros((1, 18023))
        self.bodipy_generator(list(positions.flatten()), list(substitutions.flatten()))
        descriptor = self.slatm_generator()
        return descriptor.reshape(1, -1)

        # for pos,group in zip(positions, substitutions):
        #     if int(group) != 0:
        #         descriptor[int(pos) - 1, int(group) - 1] = 1.


class BayesOpt:
    """
    Simple Bayesian Optimization routine using Expected Improvement algorithm.
    Surrogate model: Gaussian Process implemented using scikit-learn. Multiple
    evaluation EI minimization idea inspired from Martin Krasser's blog.
    """

    def __init__(self, ex_v_ex=0.01, n_restart=25):
        self.ex_v_ex = 0.01
        self.n_restart = n_restart

    def ei(self, x_query, x_prev, y_prev, gpr_model):
        """
        Get expected improvement. if sigma=0, ei = 0;
        """
        mu, sigma = gpr_model.predict(x_query, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(y_prev)
        imp = mu - mu_sample_opt - self.ex_v_ex
        Z = imp/(sigma+0.0000001)  # avoid nan
        ei = imp*sp.stats.norm.cdf(Z) + sigma * sp.stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0  # ei =0 if sigma =0
        return ei.flatten()
	# for some reason now this is causing probelem with lbfgs.
	# while initially it was not. This work for now, need to look into it deeper

    def next_location(self, x_prev, y_prev, gpr_model, constraints):
        """
        Get next possible location to evaluate the model on. 
        Iterate n_restart times to get best EI, propose the location for next
        evaluation.
        """
        dim = x_prev.shape[1]
        min_val = 1
        min_x = None

        def objective_fun(x): return self.ei(
            x.reshape(-1, dim), x_prev, y_prev, gpr_model)
        for x0 in np.random.uniform(constraints[:, 0], constraints[:, 1], size=(self.n_restart, dim)):
            res = sp.optimize.minimize(
                objective_fun, x0=x0, bounds=constraints, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
        return min_x.reshape(-1, 1)


# =======================INIT======================
# Define GP surrogate
# sigma = 26.57 = 2l^2 X for 1hot
# length_scale = (sigma/2)^0.5 = 3.64485939 X for 1hot
# rbf = gaussian_process.kernels.RBF(length_scale=3.64485939) X for 1hot

# sigma = 840.087535153 = 2l^2
# length_scale = (sigma/2)^0.5 = 20.4949693236
rbf = gaussian_process.kernels.RBF(length_scale=20.4949693236)
gpr_model = gaussian_process.GaussianProcessRegressor(kernel=rbf, alpha=0.001)

# All possible groups and positions to choose from
groups = [i for i in range(1, 47)]
positions = [i for i in range(1, 8)]

# Instantiate Bayesian and Kernel Ridge evaluation model
bo = BayesOpt(ex_v_ex=ex_v_ex, n_restart=n_restart)
kkr = KRRModel(target=target, data_dir=data_dir)

# Constraints between valid groups and positions
constraints = []
for i in range(n_groups):
    constraints.append([1, 7])
for i in range(n_groups):
    constraints.append([1, 46])
constraints = np.array(constraints)

# Initialize seeds to build initial GP model
x_prev = []
# seed the search with s random 2D
grps = np.random.choice(groups, (n_seeds, n_groups))
for i in range(n_seeds):
    tmp = np.random.choice(positions, n_groups, replace=False)
    tmp = np.insert(tmp, len(tmp), grps[i, :])
    x_prev.append(tmp)

x_prev = np.array(x_prev)
y_prev = np.zeros((n_seeds, 1))

for i in range(n_seeds):
    y_prev[i] = kkr.get_loss(kkr.gen_descriptor(x_prev[i]))

y_prev_old = -99.0

# ================MAIN LOOP===============================================
print("=================================================================")
print("ITER\tPOS\t\tGROUPS\t\tS0S1(eV)\tTarget")
print("=================================================================")
for i in range(n_iter):
    # for iteration i, obtain updated GPR model
    gpr_model.fit(x_prev, y_prev)

    # obtain next location using EI
    x_next = bo.next_location(x_prev, y_prev, gpr_model, constraints)

    unique_pos = False
    while not unique_pos:
        # if positions are not unique then discard it and generate one more
        # random position to shuffle away
        # Ideally shall replace with symmetric position, but this helps in
        # adding more random seeding data
        tmp_positions = np.squeeze(x_next[0:n_groups]).astype(int).tolist()
        if len(set(tmp_positions)) <  n_groups:
            tmp = np.random.choice(positions, n_groups, replace=False)
            tmp = np.insert(tmp, len(tmp), 
                np.random.choice(groups, n_groups))
            x_next = tmp
        else:
            unique_pos = True

    # get the loss value at proposed location
    y_next = kkr.get_loss(kkr.gen_descriptor(x_next))
    y_s0s1 = kkr.get_s0s1(kkr.gen_descriptor(x_next))
    x_next = x_next.astype(int).reshape(1, -1).squeeze()

    # if new results are improvement over previous, and valid
    # print them. If invalid (dimension of substitution reduced)
    # then skip it.
    if (y_next > y_prev_old) and (len(set(x_next[0:len(x_next)//2])) == len(x_next)//2):
        print("{:d}\t{}\t\t{}\t\t{:f}\t{:f}".format(
            i,
            " ".join(list(map(str, x_next[:len(x_next)//2]))),
            " ".join(list(map(str, x_next[len(x_next)//2:]))),
            y_s0s1,
            target
        ))
        if (np.abs(y_next - y_prev_old) < tol):
            print("Desired Tolerance Reached")
            break
        y_prev_old = y_next
    if not (len(set(x_next[0:len(x_next)//2])) == len(x_next)//2):
        continue
    x_prev = np.vstack((x_prev, x_next))
    y_prev = np.vstack((y_prev, y_next))
print("=================================================================")
