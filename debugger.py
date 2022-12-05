from matplotlib import pyplot as plt
from numpy import linspace, pi, sqrt
from pickle import load
from tqdm import tqdm
import os

from calc_boundaries import calc_avg_boundary


if __name__ == '__main__':
    if not os.path.exists('debug'):
        os.makedirs('debug')

    for file in os.listdir('debug'):
        os.remove(f'debug/{file}')

    start = 900
    end = 999

    lattice_data = load(open('outputs/lattice_data.pkl', 'rb'))
    decision_boundaries = load(open('outputs/decision_boundaries.pkl', 'rb'))
    avg_x, avg_y = calc_avg_boundary(start, end)

    lattice_length = len(lattice_data[0])

    plt.figure(figsize=(5, 5))
    plt.title("Average Boundary")
    plt.xlabel('x')
    plt.ylabel('y')
    for boundary in decision_boundaries[start:end + 1]:
        x, y = boundary['x'], boundary['y']
        plt.plot(x, y, markersize=0.1)
    plt.plot(avg_x, avg_y, markersize=4, color='black')
    plt.savefig('debug/avg_boundary.png')

    print("Generating debug data ...")
    for i in tqdm(range(start, end + 1)):
        lattice = lattice_data[i]
        boundary = decision_boundaries[i]
        x, y = boundary['x'], boundary['y']
        height_fluctuations = sqrt(x ** 2 + y ** 2) - sqrt(avg_x ** 2 + avg_y ** 2)
        thetas = linspace(0, 360, len(height_fluctuations))

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title(f"Lattice in Frame {i}")
        plt.imshow(lattice, origin='lower')

        plt.subplot(132)
        plt.title("Decision Boundary")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x, y)
        plt.plot(avg_x, avg_y)
        plt.xlim(-lattice_length // 2, lattice_length // 2)
        plt.ylim(-lattice_length // 2, lattice_length // 2)
        plt.legend(['Boundary', 'Average Boundary'])
        
        plt.subplot(133)
        plt.title("Height Fluctuations")
        plt.xlabel('$ \\theta $')
        plt.ylabel('H $\langle \\theta \\rangle $')
        plt.plot(thetas, height_fluctuations)

        plt.savefig(f'debug/boundary_{i}.png')
        plt.close()