"""
Title: Z boson

Takes in 4 user inputs. These are the start guesses for gamma_z,
gamma_ee and m_z as well as the uncertainty confidence. All .csv
files with 3 columns are then read and put into one data array.
This array is then filtered and the paramaters for m_z, gamma_z
and gamma_ee are estimated. The data is passed through an
uncertianty filter and the values of the parameters subsequently
changed until no further data points are emitted. This final data
set is then plotted along with graphs for varying paramters against
the resulting chi-squared. The values of the parameters are printed
in the console alond with tau_z, the lifetime of the boson, and
their associated uncertainties. A .png of the graph is saved
in the same folder.

As many .csv files can be inputted and filtered correctly which
which allows for higher accuracy. Moreover, estimations can be made
for different particles since the inital guesses for the parameters
can be changed. Matplotlib plots the line of best fit for the data
provided, 3 plots that show how the chi-sqaured varies with different
values for the parameters and a 3D plot that combines the chi-squared
for the mass and width of the boson. This 3D plot allows us to see
that the relationship is roughly quadratic close to the estimates
values.

There are also appropriate checks in place, such as validation checks
for initial guesses. There are also 2 places where the code will exit
cleanly if there is a problem. The code will stop if at least 1 of the
.csv files in the folder has more than 3 columns, this is a result of
the np.vstack throwing up an error if thats the case. The code will
also cleanly exit if no appropriate estimations of the parameters can
be made withing the runtime allowed in scipy.optimize. Better inital
guesses will not raise this error.

Since we are fitting 3 parameters to the data, i.e. 3 degrees of
freedom, the reduced chi-square is:

chi_red = chi / (length of data - 3)

Author: Alexander Stansfield
Date created: 14/12/2021
"""
import math
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

H_BAR = 6.5821e-25 #GeV s

while True:
    try:
        start_gamma_ee = float(input('Input the initial guess for'
                                     ' gamma_ee, research suggests'
                                     ' this is around 0.1 GeV: '))
        start_gamma_z = float(input('Input the initial guess for'
                                    ' gamma_z, research suggests this'
                                    ' is around 3 GeV*c^-2: '))
        start_m_z = float(input('Input the initial guess for m_z,'
                                ' research suggests this is'
                                ' around 90 GeV: '))
        uncertainty_confidence = int(input('Input the uncertainty confidence,'
                                           ' 3 sigma is best suited for this'
                                           ' data as there are around 100'
                                           ' data points: '))
        break
    except ValueError:
        print('Please input a number')

def general_function(energy, m_z, gamma_z, gamma_ee):
    """
    Takes in energy values and returns the cross-sectional area

    Parameters
    ----------
    E: array
    m_z: float
    gamma_z: float
    gamma_ee: float

    Returns
    -------
    1D numpy array of floats
    """
    conversion = 0.3894e6

    return (12*math.pi/(m_z**2))*(np.square(energy)/((np.square(energy) -\
              m_z**2)**2 + (m_z**2*gamma_z**2))) * gamma_ee**2 * conversion


def reduced_chi_square(observation, observation_uncertainty, prediction):
    """
    Returns the reduced chi sqaured

    Parameters
    ----------
    observation: 1D array
    observation_uncertainty: 1D array
    prediction: 1D array

    Returns:
    float
    """
    return (np.sum(((observation - prediction) / observation_uncertainty)**2))\
        / (len(observation) - 3)

def find_parameters(data):
    """
    Finds the best values for m_z, gamma_z and gamma_ee to have the
    lowest chi-square. Halts code if no optimised parameters could be
    found in the runtime allowed.

    Paramaters
    ----------
    data: 2D array of floats

    Returns
    -------
    3 floats
    """
    try:
        expected, uncertainty = scipy.optimize.curve_fit(general_function,\
                                data[:, 0], data[:, 1], sigma=data[:, 2],\
                                    p0=[start_m_z, start_gamma_z,\
                                        start_gamma_ee])
    except RuntimeError:
        print('Scipy.optimize.curve_fit was not able to find the best'
              ' parameters, please run code again and input different'
              ' starting guesses')
        sys.exit()

    return expected[0], expected[1], expected[2],\
        math.sqrt(uncertainty[0, 0]), math.sqrt(uncertainty[1, 1]),\
            math.sqrt(uncertainty[2, 2])

def create_data():
    """
    Creates the initial data array with .csv files in the folder

    Parameters
    ----------
    None

    Returns
    -------
    2D numpy array of floats
    """
    data = np.zeros((0, 3))
    print('Make sure that all files are in the same folder and have the'
          ' .csv extension')
    try:
        for filename in glob.glob('*.csv'):
            data = np.vstack((data, filter_initial(read_data(filename))))
    except ValueError:
        print('The data must have 3 columns')
        sys.exit()

    return data

def read_data(filename):
    """
    Reads in data file.

    Parameters
    ----------
    file_name : string

    Returns
    -------
    2D numpy array of floats
    """
    return np.genfromtxt(filename, dtype='float', delimiter=',', skip_header=1)

def filter_initial(data):
    """
    A filter which removes all nans, removes all <= 0 uncertainties
    and removes values which are more than 3 interquartile ranges
    outside of the lower/upper quartile

    Paramaters
    ----------
    data: 2D array of floats and nans

    Returns
    -------
    2D numpy array of floats
    """
    index = 0
    for line in data:
        for value in enumerate(line):
            if np.isnan(value[1]):
                data = np.delete(data, index, axis=0)
                index -= 1
                break
        if line[2] == 0:
            data = np.delete(data, index, axis=0)
            index -= 1
        index += 1
    lower_quartile = np.quantile(data[:, 1], 0.25, interpolation='midpoint')
    upper_quartile = np.quantile(data[:, 1], 0.75, interpolation='midpoint')
    interquartile_range = upper_quartile - lower_quartile
    data = data[~(data[:, 1] > upper_quartile + 3*interquartile_range)]
    data = data[~(data[:, 1] < lower_quartile - 3*interquartile_range)]
    return data

def find_final_parameters(data):
    """
    Finds the best parameters for function and filters data

    Parameters
    ----------
    data: 2D numpy array of floats

    Returns
    -------
    2D numpy array of floats
    3 floats
    """
    while True:
        m_z, gamma_z, gamma_ee, uncertainty_m_z, uncertainty_gamma_z,\
            uncertainty_gamma_ee = find_parameters(data)
        data, count = uncertainty_filter(data, m_z, gamma_z, gamma_ee)
        if count == 0:
            break
    tau_z = H_BAR / gamma_z
    uncertainty_tau_z = H_BAR * (1 / (gamma_z)**2) * uncertainty_gamma_z
    return data, m_z, gamma_z, gamma_ee, uncertainty_m_z, uncertainty_gamma_z,\
        uncertainty_gamma_ee, tau_z, uncertainty_tau_z

def uncertainty_filter(data, m_z, gamma_z, gamma_ee):
    """
    Removes all data which are greater than uncertainty confidence*standard
    deviation away from the prediction. Returns how many values were removed

    Parameters
    ----------
    data: 2D array of floats
    m_z: float
    gamma_z: float
    gamma_ee: float

    Returns
    -------
    2D numpy array of floats
    float
    """
    count = 0
    index = 0
    for line in data:
        if line[1] + line[2]*uncertainty_confidence <\
            general_function(line[0], m_z, gamma_z, gamma_ee)\
                or line[1] - line[2]*uncertainty_confidence >\
                    general_function(line[0], m_z, gamma_z, gamma_ee):
            data = np.delete(data, index, axis=0)
            count += 1
            index -= 1
        index += 1
    return data, count

def plot_data(data, m_z, gamma_z, gamma_ee):
    """
    Produces a 2D plot of the data.

    Parameters
    ----------
    data: 2D numpy array of floats.

    Returns
    -------
    figure
    """

    m_data = np.linspace(m_z - 10, m_z + 10, len(data[:, 0]))
    gamma_z_data = np.linspace(gamma_z - 1, gamma_z + 1, len(data[:, 0]))
    gamma_ee_data = np.linspace(gamma_ee - 0.5, gamma_ee + 0.5,\
                                len(data[:, 0]))

    chi_m_data = []
    for i in range(len(data[:, 0])):
        chi_m_data.append(reduced_chi_square(data[:, 1], data[:, 2],\
                                             general_function(data[:, 0],\
                                                              m_data[i],\
                                                                gamma_z,\
                                                                gamma_ee)))

    chi_gamma_z_data = []
    for i in range(len(data[:, 0])):
        chi_gamma_z_data.append(reduced_chi_square(data[:, 1], data[:, 2],\
                                            general_function(data[:, 0],\
                                                             m_z,\
                                                             gamma_z_data[i],\
                                                             gamma_ee)))

    chi_gamma_ee_data = []
    for i in range(len(data[:, 0])):
        chi_gamma_ee_data.append(reduced_chi_square(data[:, 1], data[:, 2],\
                                        general_function(data[:, 0],\
                                                         m_z,\
                                                         gamma_z,\
                                                         gamma_ee_data[i])))


    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 5, (1, 3))
    ax1.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='o')
    ax1.plot(np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 1000),\
             general_function(np.linspace(np.min(data[:, 0]),\
                                          np.max(data[:, 0]), 1000),\
                                                  m_z,\
                                                  gamma_z,\
                                                  gamma_ee))
    ax1.set_title('Cross-sectional area against Energy')
    ax1.set_xlabel('Energy [GeV]')
    ax1.set_ylabel('Cross-sectional area [nb]')
    ax1.legend(['Line of best fit'], loc='best')

    ax2 = fig.add_subplot(2, 5, 6)
    ax2.set_xlabel('$m_z$ [GeV$c^{-2}$]')
    ax2.set_ylabel('Reduced chi-sqaured')
    ax2.scatter(m_data, chi_m_data)
    ax2.plot(m_z, np.min(chi_m_data), 'ro')
    ax2.legend(['m_z = {:.2f}'.format(m_z)], loc='best')

    ax3 = fig.add_subplot(2, 5, 7)
    ax3.set_xlabel(r'$\Gamma_z$ [GeV]')
    ax3.set_ylabel('Reduced chi-sqaured')
    ax3.scatter(gamma_z_data, chi_gamma_z_data)
    ax3.plot(gamma_z, np.min(chi_gamma_z_data), 'ro')
    ax3.legend(['gamma_z = {:.2f}'.format(gamma_z)], loc='best')

    ax4 = fig.add_subplot(2, 5, 8)
    ax4.set_xlabel(r'$\Gamma_{ee}$ [GeV]')
    ax4.set_ylabel('Reduced chi-sqaured')
    ax4.scatter(gamma_ee_data, chi_gamma_ee_data)
    ax4.plot(gamma_ee, np.min(chi_gamma_ee_data), 'ro')
    ax4.legend(['gamma_ee = {:.2f}'.format(gamma_ee)], loc='best')

    return fig

def plot_3d(data, fig, m_z, gamma_z, gamma_ee):
    """
    Produces a 3D plot of chi-squared against varying m_z and gamma_z

    Parameters
    ----------
    data: 2D numpy array of floats
    fig: matplotlib fiure
    m_z: float
    gamma_z: float
    gamma_ee: float

    Returns
    -------
    None
    """
    ax5 = fig.add_subplot(2, 5, (4, 10), projection='3d')

    diference1 = 0.05
    difference2 = 0.05

    x_data = np.linspace(m_z + diference1, m_z - diference1, len(data[:, 0]))
    y_data = np.linspace(gamma_z + difference2, gamma_z - difference2,\
                    len(data[:, 0]))

    x_mesh, y_mesh = np.meshgrid(x_data, y_data)

    z_mesh = np.zeros((0, len(x_mesh[:, 0])))
    index = 0
    for line in y_mesh:
        temp = []
        for i in range(len(x_mesh[:, 0])):
            temp.append(reduced_chi_square(data[:, 1], data[:, 2],\
                                           general_function(data[:, 0],\
                                                            x_mesh[index, i],\
                                                                line[i],\
                                                                gamma_ee)))
        z_mesh = np.vstack((z_mesh, temp))
        index += 1

    ax5.scatter3D(x_mesh, y_mesh, z_mesh)
    ax5.set_xlim3d(m_z + diference1, m_z - diference1)
    ax5.set_ylim3d(gamma_z + difference2, gamma_z - difference2)
    ax5.set_zlim3d(np.min(z_mesh), np.max(z_mesh))
    ax5.set_title(r'Reduced chi-squared against $m_z$ and $\Gamma_z$')
    ax5.set_xlabel('$m_z$ [GeV$c^{-2}$]')
    ax5.set_ylabel(r'$\Gamma_z$ [GeV]')
    ax5.set_zlabel('Reduced chi-sqaured')

    plt.tight_layout()

    fig.savefig('plot_name.png')

    plt.show()

def main():
    """
    Main function. Executes all necessary functions and prints the results.

    Parameters
    ----------
    None

    Returns
    -------
    0
    """
    data = create_data()
    data, expected_m_z, expected_gamma_z, expected_gamma_ee, uncertainty_m_z,\
        uncertainty_gamma_z, uncertainty_gamma_ee, tau_z, uncertainty_tau_z = \
            find_final_parameters(data)
    print('The parameters that result in the lowest chi-squared for the data'
          ' is:\nThe mass of the boson is: {0:.4g} +/- {3:.1g} [GeVc^-2]\nThe'
          ' width of the boson is: {1:.4g} +/- {4:.1g} [GeV]'
          '\nThe partial width for the positron and electrong: {2:.4g} +/-'
          ' {5:.1g} [GeV]\nThe lifetime of the boson: {6:.3g} +/- '
          '{7:.1g} [seconds]\nThe final chi-square for the data is: {8:.3f}'\
              .format(expected_m_z, expected_gamma_z, expected_gamma_ee,\
                   uncertainty_m_z, uncertainty_gamma_z, uncertainty_gamma_ee,\
                    tau_z, uncertainty_tau_z,
                      reduced_chi_square(data[:, 1], data[:, 2],
                                         general_function(data[:, 0],
                                                          expected_m_z,
                                                          expected_gamma_z,
                                                          expected_gamma_ee))))
    fig = plot_data(data, expected_m_z, expected_gamma_z, expected_gamma_ee)
    plot_3d(data, fig, expected_m_z, expected_gamma_z, expected_gamma_ee)

    return 0

if __name__ == "__main__":
    main()
