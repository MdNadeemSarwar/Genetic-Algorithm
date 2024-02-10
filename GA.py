"""
Genetic Algorithm Stages:
- Create __population
- Fitness Function
- Selection
- Crossover
- Mutation

"""

import ga_functions as gaf
import pyximport

pyximport.install()

# ---Note---
# GA_func has a warning message but works as intended.

import GA_func
from numpy import Infinity
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import pandas as pd
from multiprocessing import Pool
import time

plt.style.use('dark_background')


class genetic_algorithm:
    def __init__(self, cities : list, n_generations : int, pop_size : int, cython : bool) -> None:
        self.__cities = cities
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.__population = []
        self.__fitness = []
        self.best_orders = []
        self.best_distances = []
        self.__order = []
        self.shortest_distance = Infinity
        self.__cython = cython
        self.__fitness_times = []
        self.__new_gen_times = []
        self.__roulette_times = []
        self.__crossover_times = []
        self.__mutate_times = []
        self.__order_index = count(0)
        self.fig, self.axs = plt.subplots(1 , 2)
        self.fig.set_figwidth(10)

    def __initialise(self) -> None:

        for i in range(0, len(self.__cities)):
            self.__order.append(i)

        for i in range(0, self.pop_size):
            self.__population.append(self.__order.copy())
            gaf.randomise_index(self.__population[i])

    # Could potentially cache existing order calcs
    def __calculate_fitness(self) -> None:
        start = time.time_ns()

        self.__fitness.clear()

        # Potential multi process

        for i in range(len(self.__population)):
            distance = gaf.get_distance(self.__population[i], self.__cities)
            if distance < self.shortest_distance:
                self.shortest_distance = distance
                self.best_orders.append(self.__population[i])
                self.best_distances.append(distance)
            
            self.__fitness.append(1 / (distance + 1))

        self.__normalise_fitness()

        self.__fitness_times.append((time.time_ns() - start) / 1e+9)
    

    def __calculate_one_fitness(self, order:list) -> tuple:

        distance = gaf.get_distance(order, self.__cities)
        if distance < self.shortest_distance:
            self.shortest_distance = distance
            self.best_orders.append(order)
            self.best_distances.append(distance)

        return tuple([order, 1 / (distance + 1)])


    def __calculate_fitness_multiprocess(self) -> None:
        start = time.time_ns()
        self.__fitness.clear()
        with Pool() as p:
            fitness_values = p.map(self.__calculate_one_fitness, self.__population)
        
        #Fitness_values is set to a mapped list of tuples -> (visit order, fitness score)
        #order is included in the list as the pool of workers will complete the calculations in any order
        #so there needed to be a way to identify each list item.


        self.__population.clear()
        self.__population = [result[0] for result in fitness_values]
        self.__fitness = [result[1] for result in fitness_values]
        self.__normalise_fitness()

        self.__fitness_times.append((time.time_ns() - start) / 1e+9)

    """
    ---OUTPUT FROM MULTIPROCESS TESTING---

    Generation 0 DONE
    Generation 1 DONE
    GA DURATION: 46.5628574
    Average fitness Calculation Times = 0.4887393
    Average new generation Calculation Times = 22.6854072
    Generation 0 DONE
    Generation 1 DONE
    GA MULTIPROCESS DURATION: 48.714819
    Average fitness Calculation Times = 2.08906365
    Average new generation Calculation Times = 22.15735575
    """


    def __normalise_fitness(self) -> None:
        total = sum(self.__fitness)
        self.__fitness = [n/total for n in self.__fitness]


    def __new_generation(self) -> None:
        
        new_population = []
        if self.__cython == True:
            f_loop = time.time_ns()
            self.__population = GA_func.new_gen(self.__population, self.__fitness)
            #print(f'For Loop iter time = {(time.time_ns() - f_loop) / 1e+9}')
        
        else:
            #f_loop = time.time_ns()
            for _ in range(0, len(self.__population)):
                    start = time.time_ns()
                    ideal_candidate_a = gaf.roulette_wheel_selection(self.__population, self.__fitness)
                    ideal_candidate_b = gaf.roulette_wheel_selection(self.__population, self.__fitness)
                    self.__roulette_times.append((time.time_ns() - start) / 1e+9)
                    child = gaf.ordered_crossover(ideal_candidate_a, ideal_candidate_b)
                    self.__crossover_times.append((time.time_ns() - start) / 1e+9)
                    mutated_order = gaf.mutate(child, 0.05)
                    self.__mutate_times.append((time.time_ns() - start) / 1e+9)
                    new_population.append(mutated_order)
        
            #print(f'For Loop iter time = {(time.time_ns() - f_loop) / 1e+9}')

            self.__population.clear()
            self.__population = new_population
            #print("Generation {} DONE".format(next(self.__gen_number)))
            stop = time.time_ns()
            self.__new_gen_times.append((stop - f_loop) / 1e+9)
        

    def run(self):
        self.__initialise()
        for i in range(0, self.n_generations):
            self.__calculate_fitness()
            self.__new_generation()
            print(f"Generation {i} Done", end="\r")


    def draw(self, i):
        iteration = next(self.__order_index)
     
        order_length = len(self.best_orders)

        if iteration < order_length:
            ordered___cities = []
            for i in self.best_orders[iteration]:
                ordered___cities.append(self.__cities[i])

            x_cords, y_cords = zip(*ordered___cities)

            # Tuples turned into lists and starting point is added to the end
            x_cords = list(x_cords)
            x_cords.append(x_cords[0])
            y_cords = list(y_cords)
            y_cords.append(y_cords[0])

            self.axs[0].cla()
            self.axs[1].cla()

            self.axs[0].plot(x_cords, y_cords, marker = 'o', ms = 0.5, lw = 0.1, color = 'green')
            self.axs[1].plot(np.arange(0, iteration), self.best_distances[:iteration], marker = 'o')
            self.axs[1].set_xlabel('Best Found')
            self.axs[1].set_ylabel('Distance')

        else: self.__order_index = count(0)

    def save_result(self):
        df = pd.DataFrame({'Orders' : self.best_orders, 'Distances' : self.best_distances})
        df.to_csv(f'Results/{self.n_generations}gens{self.pop_size}pop.csv')
            
    def performance_metrics(self) -> None:
        
        if len(self.__fitness_times) > 0 and len(self.__new_gen_times) > 0:
            print(f'Average fitness Calculation Times = {np.mean(self.__fitness_times)}')
            print(f'Average new generation Calculation Times = {np.mean(self.__new_gen_times)}')
            print(f'Average Roulette Calculation Times = {np.mean(self.__roulette_times)}')
            print(f'Average Crossover Calculation Times = {np.mean(self.__crossover_times)}')
            print(f'Average Mutation Calculation Times = {np.mean(self.__mutate_times)}')

            self.__metric_figure, self.__metric_axs = plt.subplots(1, 3)
            self.__metric_figure.set_figwidth(12)
            self.__metric_axs[0].plot([i for i, v in enumerate(self.__roulette_times)], 
                self.__roulette_times)
            self.__metric_axs[0].set_xlabel('Iterations')
            self.__metric_axs[0].set_ylabel('Time Taken in Seconds')
            self.__metric_axs[0].set_title('Roulette Times')
            
            self.__metric_axs[1].plot([i for i, v in enumerate(self.__crossover_times)], 
                self.__crossover_times)
            
            self.__metric_axs[1].set_xlabel('Iterations')
            self.__metric_axs[1].set_ylabel('Time Taken in Seconds')
            self.__metric_axs[1].set_title('Crossover Times')
            
            self.__metric_axs[2].plot([i for i, v in enumerate(self.__mutate_times)], 
                self.__mutate_times)

            self.__metric_axs[2].set_xlabel('Iterations')
            self.__metric_axs[2].set_ylabel('Time Taken in Seconds')
            self.__metric_axs[2].set_title('Mutation Times')


        else: print('No Data Available. Run the Algorithm')


    """
    ---OUTPUT FROM GA FUNCTION TIME TESTING---

    GA DURATION: 45.8963398
    Average fitness Calculation Times = 0.4910363
    Average new generation Calculation Times = 22.3516109
    Average Roulette Calculation Times = 10.973220573499997
    Average Crossover Calculation Times = 11.196669352999999
    Average Mutation Calculation Times = 11.1967192925
    """

#import timeit

# Cython test function not included in final version so that it can't
# be called accidentally.

"""
def ga_cython_test(n_generations, __population_size):

    #start = time.time_ns()
    #ga = GA(__cities, n_generations, __population_size, False)
    #ga.run()
    #stop = time.time_ns()
    #print(f'GA DURATION: {(stop-start) / 1e+9}')
    #ga.performance_metrics()
    #del ga
    start = time.time_ns()
    ga_multi = GA(__cities, n_generations, __population_size, True)
    ga_multi.run()
    stop = time.time_ns()
    print(f'GA CYTHON DURATION: {(stop-start) / 1e+9}')
    #ga_multi.performance_metrics()
    #print('Time: {}'.format(timeit.timeit('GA_func.new_gen(ga_multi.__population, ga_multi.fitness)', globals=globals(), number=5)))

    plt.show()

"""
