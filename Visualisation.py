import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
import re

dirname = os.path.dirname(__file__)
os.chdir(dirname)

def create_cities(x, y)-> list:
    return [tuple([x[i], y[i]]) for i in range(0, len(x) - 1)]


city_data = pd.read_csv('data.csv').iloc[:5000]

cities = create_cities(city_data.X, city_data.Y)

result_location = 'Results/500gens100pop.csv'
data = pd.read_csv(result_location)


def convert_string_int_list(string_list : list) -> list:
    num_list = re.findall("\d+", string_list)
    num_list = [int(num_list[i]) for i in range(len(num_list))]
    return num_list

best_orders = [convert_string_int_list(order) for order in data['Orders']]
best_distances = data['Distances']

import numpy as np
from itertools import count

order_index = count(0)
fig, axs = plt.subplots(1, 2)
fig.set_figwidth(14)
fig.set_figheight(6)

# Draw's i parameter is for the FuncAnimation functionallity
# and is not set to anything.
def draw(i):
    global fig
    global axs
    global order_index
    global best_orders
    global cities
    global best_distances
    global result_location
    ordered_cities = []

    iteration = next(order_index)
    
    order_length = len(data['Orders'])

    if iteration < order_length:
        ordered_cities = []
        for i in best_orders[iteration]:
            ordered_cities.append(cities[i])

        x_cords, y_cords = zip(*ordered_cities)

        # Tuples turned into lists and starting point is added to the end
        x_cords = list(x_cords)
        x_cords.append(x_cords[0])
        y_cords = list(y_cords)
        y_cords.append(y_cords[0])

        axs[0].cla()
        axs[1].cla()


        axs[0].plot(x_cords, y_cords, marker = 'o', ms = 0.5, lw = 0.1, color = 'green')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_title('Current Visit Order')
        axs[1].plot(np.arange(0, iteration), best_distances[:iteration], marker = 'o')
        axs[1].set_xlabel('Best Found')
        axs[1].set_ylabel('Distance')

        #Regex function takes the iteration count from the string for use in the title
        axs[1].set_title(f'Best Distances out of {str(re.search("[0-9]+", string=result_location).group(0))} Generations', fontsize = 11)
        axs[1].grid(True, axis='both', lw  = 0.1)
        axs[1].set_xticks(np.arange(0, iteration + 1, 1))
    else: order_index = count(0)

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def draw_animation():
    global fig
    ani = FuncAnimation(fig, draw, interval = 500)
    plt.show()
    return ani

def save_animation(ani : FuncAnimation):
    loc = r"Results/Animation.gif"
    writergif = PillowWriter(fps=1.5)
    ani.save(loc, writergif)

def main():
    save = False
    gif = draw_animation()
    if save:
        save_animation(gif)

if __name__ == '__main__':
    main()