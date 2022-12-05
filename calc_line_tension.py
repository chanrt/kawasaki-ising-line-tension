from matplotlib import pyplot as plt
from numpy import log
from pickle import load
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def calc_line_tension():
    start_k = 1
    end_k = 3
    fluctuation_spectra = load(open("outputs/fluctuation_spectra.pkl", 'rb'))

    intervals = fluctuation_spectra['intervals']
    num_intervals = len(intervals) - 1
    height_fluctuation_ft_records = fluctuation_spectra['height_fluctuations_ft']
    k_records = fluctuation_spectra['k']
    avg_perimeters = fluctuation_spectra['avg_perimeters']
    temperature = fluctuation_spectra['temperature']

    line_tensions = []
    k_b = 1

    print("Calculating Line Tension ...")
    for i in tqdm(range(num_intervals)):
        start_frame = intervals[i]
        end_frame = intervals[i + 1] - 1

        k = k_records[i]
        height_fluctuation_ft = height_fluctuation_ft_records[i]
        avg_perimeter = avg_perimeters[i]

        start_index = (abs(k - start_k)).argmin()
        end_index = (abs(k - end_k)).argmin()

        k = k[start_index:end_index]
        height_fluctuation_ft = height_fluctuation_ft[start_index:end_index]

        inverse_k_sq = 1 / k ** 2
        log_inverse_k_sq = log(inverse_k_sq)
        log_height_fluctuation_ft = log(height_fluctuation_ft)

        model = LinearRegression()
        model.fit(log_inverse_k_sq.reshape(-1, 1), log_height_fluctuation_ft)
        intercept = model.intercept_
        slope = model.coef_[0]
        line_tension = (k_b * temperature) / (avg_perimeter * slope)
        line_tensions.append(line_tension)

        plt.figure()
        plt.title("Line Tension Calculation ({} - {} frames)".format(start_frame, end_frame))
        plt.xlabel("1 / k^2")
        plt.ylabel("$ \langle | h (k) |^2 \\rangle $")
        plt.scatter(log_inverse_k_sq, log_height_fluctuation_ft)
        plt.plot(log_inverse_k_sq, intercept + slope * log_inverse_k_sq, color='red')
        plt.savefig("outputs/line_tension_{}_{}.png".format(start_frame, end_frame))

    avg_line_tensions = sum(line_tensions) / len(line_tensions)
    sd_line_tension = (sum([(line_tension - avg_line_tensions) ** 2 for line_tension in line_tensions]) / len(line_tensions)) ** 0.5

    output_string = ""
    for i in range(len(line_tensions)):
        output_string += "Line Tension ({} - {} frames): {}\n".format(intervals[i], intervals[i + 1] - 1, line_tensions[i])
    output_string += "Average Line Tension: {}\n".format(avg_line_tensions)
    output_string += "Standard Deviation of Line Tension: {}\n".format(sd_line_tension)

    with open(f"outputs/line_tension.txt", 'w') as f:
        f.write(output_string)


if __name__ == '__main__':
    calc_line_tension()