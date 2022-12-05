from matplotlib import pyplot as plt
from math import exp, floor
from numba import njit
from numpy import meshgrid, zeros
from numpy.random import random, randint
from pickle import dump
from skimage.measure import label
from sklearn.svm import SVC
from tqdm import tqdm
import os
import shutil

from calc_boundaries import calc_boundaries
from calc_fluctuations import calc_fluctuations
from calc_line_tension import calc_line_tension


@njit(fastmath=True, nogil=True)
def update(lattice):
    for i in range(length * length):
        while True:
            i1, j1, i2, j2 = get_random_pair()

            if lattice[i1, j1] != lattice[i2, j2]:
                break

        energy_before = get_energy(lattice)
        lattice[i1, j1] *= -1
        lattice[i2, j2] *= -1
        energy_after = get_energy(lattice)
        delta_energy = energy_after - energy_before

        if delta_energy < 0:
            continue
        else:
            p_exchange = exp(-beta * delta_energy)
            if random() < p_exchange:
                continue
            else:
                lattice[i1, j1] *= -1
                lattice[i2, j2] *= -1

    return lattice


@njit(fastmath=True, nogil=True)
def get_energy(lattice):
    energy = 0

    if periodic_boundary:
        for i in range(length):
            for j in range(length):
                energy += -interaction_energy * lattice[i, j] * (lattice[(i + 1) % length, j] + lattice[i, (j + 1) % length])
    else:
        for i in range(length):
            for j in range(length):
                if i != length - 1:
                    energy += -interaction_energy * lattice[i, j] * lattice[i + 1, j]
                if j != length - 1:
                    energy += -interaction_energy * lattice[i, j] * lattice[i, j + 1]

    return energy


@njit(fastmath=True, nogil=True)
def get_counts(labelled_lattice, counts):
    for i in range(2 * length):
        for j in range(2 * length):
            if labelled_lattice[i, j]:
                counts[labelled_lattice[i, j] - 1] += 1


@njit(fastmath=True, nogil=True)
def get_com(labelled_lattice, label, size):
    sum_x, sum_y = 0, 0

    for i in range(2 * length):
        for j in range(2 * length):
            if labelled_lattice[i, j] == label:
                sum_x += j
                sum_y += i

    com_x = floor(sum_x / size)
    com_y = floor(sum_y / size)

    return com_x, com_y


@njit(fastmath=True, nogil=True)
def get_random_pair():
    i1, j1 = randint(0, length, 2)
    i2, j2 = -1, -1

    if global_transport:
        while True:
            i2, j2 = randint(0, length, 2)
            if i1 != i2 and j1 != j2:
                break
    else:
        if periodic_boundary:
            neigh = randint(0, 4)
            if neigh == 0:
                i2 = i1 + 1 if i1 < length - 1 else 0
                j2 = j1
            elif neigh == 1:
                i2 = i1 - 1 if i1 > 0 else length - 1
                j2 = j1
            elif neigh == 2:
                i2 = i1
                j2 = j1 + 1 if j1 < length - 1 else 0
            else:
                i2 = i1
                j2 = j1 - 1 if j1 > 0 else length - 1
        else:
            neighs = []
            if i1 != 0:
                neighs.append((i1 - 1, j1))
            if i1 != length - 1:
                neighs.append((i1 + 1, j1))
            if j1 != 0:
                neighs.append((i1, j1 - 1))
            if j1 != length - 1:
                neighs.append((i1, j1 + 1))

            i2, j2 = neighs[randint(0, len(neighs))]
                
    return i1, j1, i2, j2


def prepare_random_lattice():
    lattice = zeros((length, length), dtype=int)

    for i in range(length):
        for j in range(length):
            if random() < 0.3:
                lattice[i, j] = 1
            else:
                lattice[i, j] = -1

    return lattice


def get_decision_boundary(lattice):
    x, y = [], []

    for i in range(length):
        for j in range(length):
            x.append((j, i))
            y.append(lattice[i, j])

    clf = SVC(kernel='rbf', gamma='auto', C=length)
    clf.fit(x, y)
    
    x_mesh, y_mesh = meshgrid(range(length), range(length))
    z = clf.decision_function(list(zip(x_mesh.ravel(), y_mesh.ravel())))
    z = z.reshape(x_mesh.shape)

    return x_mesh, y_mesh, z


def get_adjusted_lattice(lattice):
    # get expanded lattice to combat periodic boundary condition
    expanded_lattice = zeros((2 * length, 2 * length), dtype=int)

    # top row
    expanded_lattice[0:length // 2, 0:length // 2] = lattice[length - length // 2:length, length - length // 2:length]
    expanded_lattice[0:length // 2, length // 2: 3 * length // 2] = lattice[length - length // 2: length, :]
    expanded_lattice[0:length // 2, 3 * length // 2:2 * length] = lattice[length - length // 2:length, 0:length // 2]

    # middle row
    expanded_lattice[length // 2: 3 * length // 2, 0: length // 2] = lattice[:, length - length // 2: length]
    expanded_lattice[length // 2: 3 * length // 2, length // 2: 3 * length // 2] = lattice[:, :]
    expanded_lattice[length // 2: 3 * length // 2, 3 * length // 2: 2 * length] = lattice[:, 0: length // 2]

    # bottom row
    expanded_lattice[3 * length // 2:2 * length, 0:length // 2] = lattice[0:length // 2, length - length // 2:length]
    expanded_lattice[3 * length // 2: 2 * length, length // 2: 3 * length // 2] = lattice[0: length // 2, :]
    expanded_lattice[3 * length // 2:2 * length, 3 * length // 2:2 * length] = lattice[0:length // 2, 0:length // 2]

    # label the lattice
    labelled_lattice = label(expanded_lattice, connectivity=2, background=-1)
    num_labels = labelled_lattice.max()

    # get sizes of clusters
    counts = zeros(num_labels, dtype=int)
    get_counts(labelled_lattice, counts)

    # identify the biggest cluster
    biggest_cluster_label = counts.argmax() + 1
    biggest_cluster_size = counts.max()

    # isolate and center the biggest cluster
    com_x, com_y = get_com(labelled_lattice, biggest_cluster_label, biggest_cluster_size)

    if com_x < length // 2:
        com_x += length
    if com_y < length // 2:
        com_y += length

    result = expanded_lattice[com_y - length // 2: com_y + length // 2, com_x - length // 2: com_x + length // 2]

    return result


def kawasaki_ising():
    data = []
    lattice = prepare_random_lattice()
    energies = zeros(eq_time + simulation_time)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    plt.figure(figsize=(5, 5))
    plt.title("Initial lattice")
    plt.imshow(lattice, origin='lower')
    plt.savefig('outputs/initial_lattice.png')
    plt.show()    

    print("Equilibriating Kawasaki Ising model ...")
    for i in tqdm(range(eq_time)):
        lattice = update(lattice)
        energies[i] = get_energy(lattice)

    adjusted_lattice = get_adjusted_lattice(lattice)
    plt.figure(figsize=(5, 5))
    plt.title("Lattice after equilibriation")
    plt.imshow(adjusted_lattice, origin='lower')
    plt.savefig('outputs/eq_lattice.png')
    plt.show()

    print("Simulating ...") 
    for i in tqdm(range(simulation_time)):
        lattice = update(lattice)
        energies[i + eq_time] = get_energy(lattice)
        adjusted_lattice = get_adjusted_lattice(lattice)
        data.append(get_adjusted_lattice(lattice))

    with open('outputs/lattice_data.pkl', 'wb') as f:
        dump(data, f)

    x_mesh, y_mesh, z = get_decision_boundary(adjusted_lattice)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title("Final lattice after {} N^2 iterations".format(eq_time + simulation_time))
    plt.imshow(lattice, origin='lower')
    plt.subplot(132)
    plt.title("Adjusted for periodic boundary condition")
    plt.imshow(adjusted_lattice, origin='lower')
    plt.contour(x_mesh, y_mesh, z, colors='w', levels=[0], linestyles=['-'])
    plt.subplot(133)
    plt.title("Energy Variation")
    plt.plot(energies)
    plt.savefig('outputs/final_lattice.png')
    plt.show()


if __name__ == '__main__':
    print("Program started")

    length = 32
    eq_time = 5000
    simulation_time = 2000
    interaction_energy = 1
    k = 1
    temperature = 1
    beta = 1 / (k * temperature)

    global_transport = True
    periodic_boundary = True

    if os.path.exists('outputs'):
        shutil.rmtree('outputs')

    kawasaki_ising()
    calc_boundaries()
    calc_fluctuations([i * 200 for i in range(11)], temperature)
    calc_line_tension()