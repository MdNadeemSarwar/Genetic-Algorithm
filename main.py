from GA import genetic_algorithm
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import os

# Changes working directory relative to the file so it can find data.csv
directory = os.path.dirname(__file__)
os.chdir(directory)

data = pd.read_csv("data.csv")

# Creates a list of tuples that the GA can process.
def create_cities(x, y)-> list:
    return [tuple([x[i], y[i]]) for i in range(0, len(x) - 1)]

cities = create_cities(data.X[:5000], data.Y[:5000])

def main():

    start = time.time()
    ga = genetic_algorithm(cities=cities, n_generations=10, pop_size=25, cython=True)
    ga.run()
    end = time.time()
    print("GA Execution time: {0}ms".format(round(1000* (end - start), 3)))
    print("Best Distance: {0}".format(ga.best_distances[-1]))
    
    #Save results to the Results/ Folder with the filename containing
    #the number of generations and population size.
    ga.save_result()
   
    #Shows an animation of final result
    ani = FuncAnimation(ga.fig, func=ga.draw, interval = 500)
    plt.show()

    """
    USE VISUALISATION.PY TO VIEW RESULTS AFTER GA RUNTIME

    To avoid running the GA again, run visualisation.py,
    changing the filepath name (result_location) will provide different results based on the file name.
    eg. 500gens100pop OR 10gens10pop
    """

if __name__ == '__main__':
    main()