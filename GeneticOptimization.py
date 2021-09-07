from operator import sub
import numpy as np
from copy import deepcopy
import random


class GA:
    """
    This module contains collections of all functions and routines
    needed for genetic algorithm based optimization.
    It will contain namely:
    evaluation : takes in list of descriptors and evaluate them one
                by one
    recombination : picks up the best one and keep them in a sinle
                element in class
    mutation : mutate parents and saves in elements to be evaluated
    crossover : generate next generation
    """
    def __init__(self,
                mutation_rate = 0.01,
                population_size = 10,
                max_generation = 10,
                dimensions=7,
                max_eval=10,
                obj_func = None,
                target=0.0):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.current_population = []
        self.next_generation = []
        self.to_evaluate = []
        self.group_size = 46
        self.current_loss = 999.0
        self.current_best_loss = 0.0
        self.max_generation = max_generation
        self.max_eval = max_eval
        self.obj_func = obj_func
        self.dimensions = dimensions
        self.target = target

    def objective_func(self,bit_field):
        """
        inputs bitstring and return the function value
        """
        # print("obfun",bit_field)
        input_fields = self._bit_decode2(bit_field)
        # print(self._bit_decode2(bit_field))
        loss = self.obj_func(input_fields)
        return loss

    def evaluate(self):
        """
        evaluate all the molecules to be evaluated. Saved in 
        list to_evaluate.
        """
        i = 0
        j = len(self.to_evaluate)
        for input_bits in self.to_evaluate:
            evaluated_loss = self.objective_func(input_bits)
            print('Evaluating: {} of {}   \r'.format(i+1,j), end="")
            if evaluated_loss < self.current_loss:
                self.next_generation.append([input_bits, evaluated_loss])
            i = i + 1
        print("")


    def recombine(self):
        tmp_population = []
        tmp_population.extend(self.current_population)
        tmp_population.extend(self.next_generation)
        key_fn = lambda x: x[1]
        tmp_population = sorted(tmp_population,key=key_fn)
        self.current_population = deepcopy(tmp_population[0:self.population_size])
        self.to_evaluate = []
        self.next_generation = []

    def mutate(self,bit_field):
        """
        go over each element and flip bit if probability is less then 
        self.mutation_rate
        """
        mat = np.array(bit_field).reshape(7,-1)
        if np.random.uniform() < self.mutation_rate:
            row = np.random.choice(range(7))
            mat[row,:] = 0
            mat[row, np.random.choice(range(self.group_size))] = 1
        bit_field = mat.reshape(1, -1).squeeze().tolist()
        return bit_field

    def crossover(self,bit_field1, bit_field2):
        "Randomly take elements from two bitfield and generate "
        mat1 = np.array(bit_field1).reshape(7,-1)
        mat2 = np.array(bit_field2).reshape(7,-1)
        mat3 = np.zeros(mat1.shape)
        for i in range(mat1.shape[0]):
            mat3[i, :] = mat1[i, :] if (
                np.random.uniform() < 0.5) else mat2[i, :]
        bit_field = mat3.reshape(1, -1).squeeze().tolist()
        return bit_field

    def populate(self):
        groups = [i for i in range(1, 47)]
        positions = [i for i in range(1, 8)]
        x_prev = []
        # seed the search with s random 2D
        print("Starting population estimation")
        grps = np.random.choice(groups, (self.population_size, self.dimensions))
        for i in range(self.population_size):
            tmp = np.random.choice(positions, self.dimensions, replace=False)
            tmp = np.insert(tmp, len(tmp), grps[i, :])
            x_prev.append(tmp)
            # print(tmp)

        x_prev = np.array(x_prev)
        y_prev = np.zeros((self.population_size, 1))
        # print(x_prev)
        # exit()
        for i in range(self.population_size):
            sub_array = x_prev[i]
            # print(sub_array)
            pos_sub = sub_array[0:len(sub_array)//2].astype(int)
            grp_sub = sub_array[len(sub_array)//2:len(sub_array)].astype(int)
            # print(pos_sub, grp_sub)
            y_prev[i] = self.objective_func(self._bit_encode(pos_sub, grp_sub))
            self.current_population.append(
                [self._bit_encode(pos_sub, grp_sub), y_prev[i]])
            print('Calculating parent: {}   \r'.format(i+1), end="")
        print("")
        key_fn = lambda x: x[1]
        self.current_population = sorted(self.current_population, key=key_fn)

    def population_selection(self):
        pass

    def update_score(self):
        losses = []
        for parent in self.current_population:
            try:
                losses.append(parent[1][0])
            except IndexError:
                losses.append(parent[1])
        self.current_loss = np.median(np.array(losses))
        self.current_best_loss = losses[0]
        # print(np.median(np.array(losses)))
        # print(losses)

    def _bit_encode(self,positions, substitutions):
        mat = np.zeros((7,self.group_size))
        for pos, grp in zip(positions, substitutions):
            mat[pos - 1, grp - 1] = 1
        bit_field = mat.reshape(1, -1).squeeze().tolist()
        return bit_field

    def _bit_decode(self, bit_field):
        pos = []
        grp = []
        mat = np.array(bit_field).reshape(7, -1)
        for i, row in enumerate(mat):
            pos.append(i)
            grp.append(np.sum((np.array(range(self.group_size)) + 1) * row))
        return pos, grp
    
    def _bit_decode2(self, bit_field):
        pos = []
        grp = []
        mat = np.array(bit_field).reshape(7, -1)
        for i, row in enumerate(mat):
            pos.append(float(i+1))
            grp.append(np.sum((np.array(range(self.group_size)) + 1) * row))
        # print("decode", pos, (grp))
        pos.extend(grp)
        return np.array(pos)

    def print_iter(self,generation):
        print("Current Gen {:d}, Median: {:f}  Best: {:f}"
                .format(generation,
                    self.target + (self.current_loss)**0.5,
                    self.target + (self.current_best_loss)**0.5))
        pos,grp = self._bit_decode(self.current_population[0][0])
        print("Best Groups {} ; Pos {}".format(grp, pos))
        # except TypeError:
        #     print("TypeErr", generation, self.current_loss,
        #             self.current_best_loss)


    def optimize(self,tol):
        """
        optimization in GA
        """
        # populate the corpus
        self.populate()
        # randomly select 0.25 to 1.00 times of current population for reproduction
        next_parents = self.current_population[0:int(np.random.uniform(
            low=self.population_size/4,
            high=self.population_size))]
        num_parents = len(next_parents)
        self.update_score()
        self.print_iter(0)
        for generation in range(self.max_generation):
            # generate unique parent pair for children
            for i in range(0, num_parents):
                for j in range(i + 1, num_parents):
                    # print(next_parents[i], next_parents[j])
                    child = self.crossover(
                        next_parents[i][0], next_parents[j][0])
                    child = self.mutate(child)
                    self.to_evaluate.append(child)
            random.shuffle(self.to_evaluate)
            # Limit evaluations
            self.to_evaluate = self.to_evaluate[0:np.max(self.max_eval, 0)]
            # evaluate children
            self.evaluate()
            # recombine population
            self.recombine()
            # update scores
            self.update_score()
            if np.abs(self.current_best_loss - self.target) < tol:
                print("Desired Tolerance Reached")
                self.print_iter(generation + 1)
                break
            self.print_iter(generation + 1)
