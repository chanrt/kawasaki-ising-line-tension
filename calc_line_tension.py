from matplotlib import pyplot as plt
from numpy import log
from pickle import load
from sklearn.linear_model import LinearRegression


def calc_line_tension(k_b, temperature):
    start_k = 1
    end_k = 3
    fluctuation_spectra = load(open("outputs/fluctuation_spectra.pkl", 'rb'))

    intervals = fluctuation_spectra['intervals']
    num_intervals = len(intervals) - 1
    height_fluctuation_ft_records = fluctuation_spectra['height_fluctuations_ft']
    k_records = fluctuation_spectra['k']
    avg_perimeters = fluctuation_spectra['avg_perimeters']

    print("Calculating Line Tension ...")
    for i in range(num_intervals):
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
        print(f"Line Tension between frames {start_frame} and {end_frame}: {line_tension}")

        plt.figure()
        plt.title("Line Tension Calculation ({} - {} frames)".format(start_frame, end_frame))
        plt.scatter(log_inverse_k_sq, log_height_fluctuation_ft)
        plt.plot(log_inverse_k_sq, intercept + slope * log_inverse_k_sq, color='red')
        plt.savefig("outputs/line_tension_{}_{}.png".format(start_frame, end_frame))


if __name__ == '__main__':
    calc_line_tension(1, 1)