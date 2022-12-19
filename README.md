# kawasaki-ising-line-tension

Calculating line tension using phase boundary fluctuation spectra in Ising model following Kawasaki dynamics.
This project was done for the course PH322: Molecular Simulations, offered at Indian Institute of Science (Bangalore) in fall semester of 2022.

Set length (N), equilibration time (in units of N^2), simulation time (in units of N^2), interaction energy (J) and temperature (T) in the main.py file.
You can enable/disable global transport and periodic boundary condition.
You need to specify the intervals across which averaging should take place.

The main.py program will:
1) Adjust for periodic boundary condition (if that is the case)
2) Draw phase boundaries using SVMs (this step is multithreaded)
3) Calculate height fluctuations with respect to the average
4) Obtain its power density spectrum and fit on the specified portion of the graph, to calculate line tension
5) Line tension will be presented to you in mean +/- sd form, in the outputs/line_tension.txt file

![line_tension_vs_temperature](https://user-images.githubusercontent.com/61446958/208427012-aad03acb-7667-4c7a-bd52-87a1f2cf362d.png)
