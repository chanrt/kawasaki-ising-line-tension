from matplotlib import pyplot as plt
from math import sqrt
from multiprocessing import Pool, cpu_count
from numpy import arctan2, cos, linspace, mean, meshgrid, pi, roll, sin, zeros
from pickle import dump, load
from scipy.interpolate import splprep, splev
from sklearn.svm import SVC
from tqdm import tqdm


def calc_avg_boundary(start, end):
    # naive formulation
    num_frames = end - start + 1
    
    boundaries = load(open('outputs/decision_boundaries.pkl', 'rb'))
    boundary_length = len(boundaries[0]['x'])
    sum_x = zeros(boundary_length)
    sum_y = zeros(boundary_length)

    for i in range(start, end + 1):
        boundary = boundaries[i]
        sum_x += boundary['x']
        sum_y += boundary['y']

    avg_x = sum_x / num_frames
    avg_y = sum_y / num_frames

    return avg_x, avg_y

    # radial formulation (unstable)
    # num_frames = end - start + 1
    # boundaries = load(open('outputs/decision_boundaries.pkl', 'rb'))
    # boundary_length = len(boundaries[0]['x'])
    # radial_averages = zeros(boundary_length)
    # thetas = linspace(0, 2 * pi, boundary_length)

    # for i in range(start, end + 1):
    #     boundary = boundaries[i]
    #     x, y = boundary['x'], boundary['y']
    #     angles = arctan2(y, x)
        
    #     for theta in thetas:
    #         # find the closest angle
    #         angles_diff = abs(angles - theta)
    #         min_index = angles_diff.argmin()
    #         radial_averages[min_index] += sqrt(x[min_index] ** 2 + y[min_index] ** 2)

    # radial_averages /= num_frames
    # avg_x, avg_y = radial_averages * cos(thetas), radial_averages * sin(thetas)

    # return avg_x, avg_y


def get_path(lattice):
    length = len(lattice)
    x, y = [], []

    # 'x' contains (x, y) coordinates, and 'y' contains the orientation
    for i in range(length):
        for j in range(length):
            x.append((j, i))
            y.append(lattice[i, j])

    # classify the points
    clf = SVC(kernel='rbf', gamma='auto', C=length)
    clf.fit(x, y)
    
    # get the decision boundary
    x_mesh, y_mesh = meshgrid(range(length), range(length))
    z = clf.decision_function(list(zip(x_mesh.ravel(), y_mesh.ravel())))
    z = z.reshape(x_mesh.shape)

    # get the contour
    contour = plt.contour(x_mesh, y_mesh, z, colors='k', levels=[0], linestyles=['-'])

    # consider the longest path
    paths = contour.collections[0].get_paths()
    path_lengths = [len(path.vertices) for path in paths]
    longest_path = paths[path_lengths.index(max(path_lengths))]
    x, y = longest_path.vertices.T

    # adjust center of mass to (0, 0)
    x = x - mean(x)
    y = y - mean(y)

    # start curve from point closest to theta = 0
    angles = abs(arctan2(y, x))
    min_angle = min(angles)
    start_index = angles.tolist().index(min_angle)
    roll(x, -start_index)
    roll(y, -start_index)

    # smooth and interpolate the path
    tck, u = splprep([x, y], s=0)
    u_fine = linspace(0, 1, 1000)
    x_fine, y_fine = splev(u_fine, tck)

    return {'x': x_fine, 'y': y_fine}


def calc_boundaries():
    num_processes = cpu_count() - 2
    data = load(open('outputs/lattice_data.pkl', 'rb'))

    print("Calculating decision boundaries ...")
    pool = Pool(num_processes)
    decision_boundaries = list(tqdm(pool.imap(get_path, data), total=len(data)))

    with open('outputs/decision_boundaries.pkl', 'wb') as f:
        dump(decision_boundaries, f)