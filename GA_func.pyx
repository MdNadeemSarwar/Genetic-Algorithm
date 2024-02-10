import random
import cython

# ---Note---
# 
# This file contains the same functions as ga_functions.py
# but is compiled by cython.
# 
#
# Descriptions of the functions can be found in ga_functions.py
#
# When imported in GA.py, it does raise a warning which can be ignored


def find_fittest(probability, random_value, population):
    rolling_total = 0
    for i in range(0, len(probability)):

        rolling_total += probability[i]
        if rolling_total > random_value:

            return population[i]

# Ran a test to see if cythons for loop ran faster which it did

"""
def cython_test():
    var = 0
    for i in range(0, 100):
        var += i / (i + 1)
"""

def roulette_wheel_selection(population : list, probability : list) -> list:
    
    probability_sum = sum(probability)
    random_value = random.uniform(0, probability_sum)
    return find_fittest(probability, random_value, population)

def new_gen(population, fitness):
    pop_len = len(population)
    new_population = []
    for _ in range(0, pop_len):
        ideal_candidate_a = roulette_wheel_selection(population, fitness)
        ideal_candidate_b = roulette_wheel_selection(population, fitness)
        child = ordered_crossover(ideal_candidate_a, ideal_candidate_b)
        mutated_order = mutate(child, 0.05)
        new_population.append(mutated_order)

    return new_population

def ordered_crossover(parent_a : list, parent_b : list) -> list:
    slice_start = random.randrange(0, len(parent_a) - 1)
    slice_end = random.randrange(slice_start + 1, len(parent_a))
    child_slice = parent_a[slice_start:slice_end]
    for location in parent_b:
        if child_slice.count(location) == 0:
            child_slice.append(location)

    return child_slice

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