import random
from math import pow, sqrt

import pyximport

pyximport.install()

import GA_func

random.seed(42)


def generate_square_coordinates(min : int, max : int, n_cities : int):
    x_cords = [random.randint(min, max) for _ in range(n_cities)]
    y_cords = [random.randint(min, max) for _ in range(n_cities)]
    population = []
    for i in range(n_cities):
        population.append(tuple([x_cords[i], y_cords[i]]))

    return population


def __euclidean_distance(point_a : tuple, point_b : tuple):
    return sqrt(pow(point_a[0] - point_b[0], 2) + pow(point_a[1] - point_b[1], 2))

# ---Get Distance---
#
# Finds the distance


def get_distance(order : list, cities : list):
    total_distance = 0
    for i in range(0, len(order) - 1):
        total_distance += __euclidean_distance(
            cities[order[i]],
            cities[order[i + 1]]
            )

    # Add Distance from end point back to start
    total_distance += __euclidean_distance(cities[order[0]], cities[order[-1]])

    return total_distance


# ---Randomise Index---
#
# Will shuffle all items in a list.
# Made to simplify the usage of random.shuffle.

def randomise_index(index_list : list):
    return random.shuffle(index_list)

# ---Roulette Wheel Selection---
#
# This function will generate a random number between 0 and 1 and
# will add the normalised fitness probabilities until one of them
# exceeds the random value. The larger the probability, the more
# likely it is that candidate will be chosen by chance.
#
# Note that larger population sizes may introduce the risk that
# the normalised probabilities are so equally distributed that
# an ideal candidate could be missed.

def roulette_wheel_selection(population : list, probability : list) -> list:
    """
    - Calculate S = the sum of the fitnesses.
    - Generate a random number between 0 and S.
    - Starting from the top of the population, keep adding the fitnesses to the partial sum P, till P<S.
    - The individual for which P exceeds S is the chosen individual.
    """
    probability_sum = sum(probability)
    random_value = random.uniform(0, probability_sum)
    rolling_total = 0
    for i in range(0, len(probability)):
        # Random point has higher chance to be in a section with a larger probability
        # Candidates with smaller probability will have little effect on the sum
        # causing it the bypass that individial. 
        
        rolling_total += probability[i]
        if rolling_total > random_value:
            #candidate found by exceeding random point
            return population[i]


# ---Swap---
#
# Used in the mutate function to swap 2 genes in the sequence
# if a certain condition has been met.

def __swap(order : list, index_a : int, index_b : int):
    index_a_value = order[index_a]
    index_b_value = order[index_b]
    new_order = []

    for i in range(len(order)):
        if i == index_a:
            new_order.append(index_b_value)
        elif i == index_b:
            new_order.append(index_a_value)
        else: new_order.append(order[i])

    return new_order

# ---Mutate---
#
# This function will randomly mutate 2 genes by swapping them if
# a randomly generated float between 0 and 1 is less than the
# mutation rate. A typical mutation rate lies between 1% and 5%.

def mutate(order : list, mutation_rate : float) -> list:
    if random.uniform(0.0, 1.0) < mutation_rate:
        index_a = random.randint(0, len(order) - 1)
        index_b = random.randint(0, len(order) - 1)
        if index_b == index_a:
            while index_b == index_a:
                index_b = random.randint(0, len(order) - 1)

        swap_order = __swap(order, index_a, index_b)

        return swap_order
    
    else: return order

# ---Single Point Crossover---
#
# This function was initially used in the development of the GA
# but introduced some bugs.
# The function did not take into account the possibility of some genes in
# the first slice of parent_a being present in parent_b, which removed some
# some genes and duplicated others. After a few iterations, all the genes were
# deleted which caused the GA to raise errors because the list of genes was empty.

def single_point_crossover(parent_a : list, parent_b: list) -> list:
    crossover_point = random.randrange(0, len(parent_a))
    child_a = parent_a[:crossover_point] + parent_b[crossover_point:]
    child_b = parent_b[:crossover_point] + parent_a[crossover_point:]
    return child_a, child_b

# ---Ordered Crossover---
# 
# This function uses the OX1 ordering method.
# Supplimentary information that was used in the
# making of this function can be found at
# https://mat.uab.cat/~alseda/MasterOpt/GeneticOperations.pdf
#
# This function takes a slice of parent_a and fills the remaining
# spaces with non-duplicate genes in parent_b

def ordered_crossover(parent_a : list, parent_b : list) -> list:
    slice_start = random.randrange(0, len(parent_a) - 1)
    slice_end = random.randrange(slice_start + 1, len(parent_a))
    child_slice = parent_a[slice_start:slice_end]
    for location in parent_b:
        if child_slice.count(location) == 0:
            child_slice.append(location)

    return child_slice


# ---Cython Roulette Wheel Selection---
#
# Slightly modified version of the existing roulette
# wheel selection.
#
# This function calls on the find fittest function from
# the pyx file which was introduced to potentially decrease
# the search time for the potential best candidate in the For loop


def cython_roulette_wheel_selection(population : list, probability : list) -> list:
    
    probability_sum = sum(probability)
    random_value = random.uniform(0, probability_sum)
    return GA_func.find_fittest(probability, random_value, population)