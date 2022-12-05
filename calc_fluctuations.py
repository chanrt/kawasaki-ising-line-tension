from matplotlib import pyplot as plt
from numpy import array, arctan2, pi, sqrt, zeros
from numpy.fft import rfft
from pickle import dump, load
from tqdm import tqdm

from calc_boundaries import calc_avg_boundary


def calc_fluctuations(intervals, temperature):
    boundaries = load(open("outputs/decision_boundaries.pkl", 'rb'))
    avg_perimeters = []
    height_fluctuation_ft_records = []
    k_records = []

    print("Calculating fluctuations ...")
    for i in tqdm(range(len(intervals) - 1)):
        start = intervals[i]
        end = intervals[i + 1] - 1
        num_frames = end - start + 1

        avg_x, avg_y = calc_avg_boundary(start, end)
        avg_angles = arctan2(avg_y, avg_x)
        avg_perimeter = 0
        for i in range(len(avg_x) - 1):
            avg_perimeter += sqrt((avg_x[i + 1] - avg_x[i]) ** 2 + (avg_y[i + 1] - avg_y[i]) ** 2)
        avg_perimeters.append(avg_perimeter)

        height_fluctuations_ft = [0]

        for frame in range(start, end + 1):
            # naive formulation
            # boundary = boundaries[frame]
            # x, y = boundary['x'], boundary['y']
            # height_fluctuations = sqrt(x ** 2 + y ** 2) - sqrt(avg_x ** 2 + avg_y ** 2)
            # fourier_transform = abs(rfft(height_fluctuations) / len(height_fluctuations)) ** 2
            # height_fluctuations_ft += fourier_transform

            # radial formulation
            boundary = boundaries[frame]
            x, y = boundary['x'], boundary['y']
            angles = arctan2(y, x)
            height_fluctuations = []

            for i, angle in enumerate(angles):
                angles_diff = abs(avg_angles - angle)
                min_index = angles_diff.argmin()
                height_fluctuation = sqrt(x[i] ** 2 + y[i] ** 2) - sqrt(avg_x[min_index] ** 2 + avg_y[min_index] ** 2)
                height_fluctuations.append(height_fluctuation)

            fourier_transform = abs(rfft(height_fluctuations) / len(height_fluctuations)) ** 2
            height_fluctuations_ft += fourier_transform

        height_fluctuations_ft /= num_frames
        k = array([2 * pi * num / avg_perimeter for num in range(len(height_fluctuations_ft))])
        
        height_fluctuation_ft_records.append(height_fluctuations_ft)
        k_records.append(k)

    plt.title("Fluctuation Spectra")
    plt.xlabel('k')
    plt.ylabel('$ \langle | h (k) |^2 \\rangle $')
    for i in range(len(intervals) - 1):
        k = k_records[i]
        height_fluctuations_ft = height_fluctuation_ft_records[i]
        plt.loglog(k, height_fluctuations_ft, label=f"Frame {intervals[i]} - {intervals[i + 1] - 1}")
    plt.legend()
    plt.savefig(f'outputs/fluctuation_spectra.png')
    plt.show()

    fluctuation_spectra = {
            'k': k_records, 
            'height_fluctuations_ft': height_fluctuation_ft_records, 
            'intervals': intervals, 
            'avg_perimeters': avg_perimeters,
            'temperature': temperature,
        }
    dump(fluctuation_spectra, open('outputs/fluctuation_spectra.pkl', 'wb'))


if __name__ == '__main__':
    intervals = [0, 200, 400, 600, 800, 1000]
    calc_fluctuations(intervals)