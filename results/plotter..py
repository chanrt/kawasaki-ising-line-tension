from matplotlib import pyplot as plt
from numpy import array
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    temperatures = array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
    line_tensions = [0.002189771, 0.004271235, 0.006650391, 0.009094335, 0.012798927, 0.016412392, 0.021005156]
    errors = [0.000140233, 0.000475937, 0.000574483, 0.000304101, 0.001696069, 0.002433238, 0.004732405]

    model = LinearRegression()
    model.fit([[temperature] for temperature in temperatures], line_tensions)
    slope = model.coef_[0]
    intercept = model.intercept_
    print("Fitted params:", slope, intercept)
    print("R^2:", model.score([[temperature] for temperature in temperatures], line_tensions))

    plt.figure()
    plt.title("Variation of Line Tensions with Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Line Tension")
    plt.errorbar(temperatures, line_tensions, yerr=errors, fmt='o', label="data")
    plt.plot(temperatures, intercept + slope * temperatures, color='red', label="linear fit with R^2 = 0.98")
    plt.legend()
    plt.savefig("results/line_tension_vs_temperature.png")
    plt.show()
    