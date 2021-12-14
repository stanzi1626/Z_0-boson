"""
Title: Bouncing Ball

This code takes in 3 user inputs: efficiency, initial height and the height
of interest.A ball instance is then created and with those parameters,
the resultant behaviour of the ball is modelled within the class. The number
of bounces is calculated precisely outside of the class since class variables
can be imprecise.
The x position of the ball is also stored in the ball class, so there i
s possibilty for the x velocity to be
manipulated and x position to be plotted.

Author: Alexander Stansfield
Date created: 19/10/2021
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

while True:
    try:
        start_gamma_ee = float(input(print('Input the initial guess for'
                                           ' gamma_ee, research suggests'
                                           ' this is around 0.1: ')))
        start_gamma_z = float(input(print('Input the initial guess for'
                                          ' gamma_z, research suggests this'
                                            ' is around 3: ')))
        start_m_z = float(input(print('Input the initial guess for m_z,'
                                      ' research suggests this is'
                                        ' around 90: ')))
        uncertainty_confidence = int(input(\
                               print('Input the uncertainty confidence, 3'
                                     ' sigma is best suited for this data'
                                         'as there are around 100 data'
                                             'points: ')))
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
        / len(observation - 1)

def find_parameters(data):
    """
    Finds the best values for m_z, gamma_z and gamma_ee to have the
    lowest chi-square

    Paramaters
    ----------
    data: 2D array of floats

    Returns
    -------
    3 floats
    """
    expected, uncertainty = scipy.optimize.curve_fit(general_function,\
                             data[:, 0], data[:, 1], sigma=data[:, 2],\
                                 p0=[start_m_z, start_gamma_z,\
                                     start_gamma_ee])
    return expected[0], expected[1], expected[2],\
        math.sqrt(uncertainty[0, 0]), math.sqrt(uncertainty[1, 1]),\
            math.sqrt(uncertainty[2, 2])

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
    A filter which removes all nans, removes all 0 uncertainties
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
        for i in range(0, len(line)):
            if np.isnan(line[i]):
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

def create_data():
    """
    Creates the initial data array with files that user inputs

    Parameters
    ----------

    Returns
    -------
    2D numpy array of floats
    """
    data = np.zeros((0, 3))
    while True:
        try:
            number_of_files = int(input(print('How many files would you like'
                                              ' to read in: ')))
            break
        except ValueError:
            print('Please enter an integer')
    for i in range(number_of_files):
        filename = input(print('Input the name of the file: '))
        temp = filter_initial(read_data(filename))
        data = np.vstack((data, temp))

    return data

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
    return data, m_z, gamma_z, gamma_ee, uncertainty_m_z, uncertainty_gamma_z,\
        uncertainty_gamma_ee

def plot_data(data, m_z, gamma_z, gamma_ee):
    """
    Produces a 2D plot of the data.

    Parameters
    ----------
    data: 2D numpy array of floats.

    Returns
    -------
    None
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
    ax2.set_xlabel('m_z [GeV$c^{-2}$]')
    ax2.set_ylabel('Reduced chi-sqaured')
    ax2.scatter(m_data, chi_m_data)
    ax2.plot(m_z, np.min(chi_m_data), 'ro')
    ax2.legend(['m_z = {:.2f}'.format(m_z)], loc='best')

    ax3 = fig.add_subplot(2, 5, 7)
    ax3.set_xlabel('gamma_z [GeV]')
    ax3.set_ylabel('Reduced chi-sqaured')
    ax3.scatter(gamma_z_data, chi_gamma_z_data)
    ax3.plot(gamma_z, np.min(chi_gamma_z_data), 'ro')
    ax3.legend(['gamma_z = {:.2f}'.format(gamma_z)], loc='best')

    ax4 = fig.add_subplot(2, 5, 8)
    ax4.set_xlabel('gamma_ee [GeV]')
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
    ax5.set_title('Reduced chi-squared against m_z and gamma_z')
    ax5.set_xlabel('m_z [GeV$c^{-2}$]')
    ax5.set_ylabel('gamma_z [GeV]')
    ax5.set_zlabel('Reduced chi-sqaured')

    plt.tight_layout()

    fig.savefig('plot_name.png')

    plt.show()

def main():
    """
    blah blah blah
    """
    data = create_data()
    data, expected_m_z, expected_gamma_z, expected_gamma_ee, uncertainty_m_z,\
        uncertainty_gamma_z, uncertainty_gamma_ee = \
            find_final_parameters(data)
    print('The parameters that result in the lowest chi-squared for the data'
          ' is:\nm_z = {0:.2f} +/- {3:.1g} [GeVc^-2]\ngamma_z = {1:.2f} +/- {4:.1g} [GeV]'
          '\ngamma_ee = {2:.3f} +/- {5:.1g} [GeV]'\
              .format(expected_m_z, expected_gamma_z, expected_gamma_ee,\
                   uncertainty_m_z, uncertainty_gamma_z, uncertainty_gamma_ee))
    fig = plot_data(data, expected_m_z, expected_gamma_z, expected_gamma_ee)
    plot_3d(data, fig, expected_m_z, expected_gamma_z, expected_gamma_ee)

    return 0

if __name__ == "__main__":
    main()
