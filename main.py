import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize

FILE_NAME1 = 'z_boson_data_1.csv'
FILE_NAME2 = 'z_boson_data_2.csv'

#line 18 data 1 86.5354,1.1222,0.0754

gamma_ee = 0.08391 #GeV
start_gamma_z = 3 #Gev
start_m_z = 90 #Gev/c^2 #values should be around these
STEP_SIZE = 0.1
TOLERANCE = 0.000001
MAX_ITERATIONS = 10000

def general_function(E, m, gamma):
    """
    Takes in energy values and returns the cross-sectional area

    Parameters
    ----------
    E: array
    m_z: float
    gamma_z: float

    Returns
    -------
    2D numpy array of floats    
    """
    conversion = 0.3894e6 #change the values into nb

    return (12*math.pi/(m**2))*(np.square(E)/((np.square(E) - m**2)**2 + (m**2*gamma**2))) * gamma_ee**2 * conversion


#the idea at the momement is to do each parameter one at a time until the chi-sqauared is within some tolerance, this 
#is a work around while the scipy.optimize isnt working

def hill_climbing(function, m_minimum, gamma_minimum, step1=STEP_SIZE, step2=STEP_SIZE):
    """
    Performs 1D hill climbing algorithm with varying step size.

    Parameters
    ----------
    function : function of single argument (float) that returns a float
    x_minimum : float, optional
        The default is START_VALUE.
    step : float, optional
        The default is STEP_SIZE.

    Returns
    -------
    x_minimum : float
        Optimum value of parameter
    minimum : float
        Minimum value of function
    counter : int
        Number of iterations
    """
    difference = 1
    minimum = function(m_minimum, gamma_minimum)
    counter = 0
    counter_1 = 0
    counter_2 = 0

    while difference > TOLERANCE:
        counter += 1

        while True:
            counter_1 += 1
            minimum1_test_minus = function(m_minimum - step1, gamma_minimum)
            minimum1_test_plus = function(m_minimum + step1, gamma_minimum)
            if minimum1_test_minus < minimum:
                m_minimum -= step1
                difference = minimum - minimum1_test_minus
                minimum = function(m_minimum, gamma_minimum)  
                break   
            elif minimum1_test_plus < minimum:
                m_minimum += step1
                difference = minimum - minimum1_test_plus
                minimum = function(m_minimum, gamma_minimum)
                break
            else:
                step1 = step1 * 0.1
            if counter_1 == 1000:
                break

        while True:
            counter_2 += 1
            minimum2_test_minus = function(m_minimum, gamma_minimum - step2)
            minimum2_test_plus = function(m_minimum, gamma_minimum + step2)
            if minimum2_test_minus < minimum:
                gamma_minimum -= step2
                difference = minimum - minimum2_test_minus
                minimum = function(m_minimum, gamma_minimum)
                break
            elif minimum2_test_plus < minimum:
                gamma_minimum += step2
                difference = minimum - minimum2_test_plus
                minimum = function(m_minimum, gamma_minimum)
                break
            else:
                step2 = step2 * 0.1
            if counter_2 == 1000:
                break

        if counter == MAX_ITERATIONS:
            print('Failed to find best solution after {0:d} iterations.'.
                  format(counter))
            break

    return m_minimum, gamma_minimum, minimum

def chi_square(observation, observation_uncertainty, prediction):
    """
    Returns the chi sqaured
    """
    return np.sum((observation - prediction)**2 / observation_uncertainty**2)

def find_parameters(data):
    """
    
    """
    x, y = scipy.optimize.curve_fit(general_function, data[:,0], data[:,1], sigma = data[:, 2], p0=[start_m_z, start_gamma_z])
    return x[0], x[1]

def read_data(filname):
    """
    Reads in data file.

    Parameters
    ----------
    file_name : string

    Returns
    -------
    2D numpy array of floats
    """
    return np.genfromtxt(filname, dtype='float', delimiter=',', skip_header=1)

def filter(data):
    """
    removes all nans
    removes all 0 uncertainties
    removes values that are 10* more than the average without it 
    """
    index = 0
    average = np.average(data[:,1])
    for line in data:
        for i in range(0, len(line)):
            if np.isnan(line[i]):
                data = np.delete(data, index, axis=0)
                index -= 1
                break
        if line[2] == 0:
            data = np.delete(data, index, axis=0)
            index -= 1
        if np.abs(((average - line[1]/len(data[:,1]))) - average) > 0.1:
            data = np.delete(data, index, axis=0)
            index -= 1
        index += 1
    return data


def plot_data(data, m_z, gamma_z):
    """
    Produces a plot of the data.

    Parameters
    ----------
    data : 2D numpy array of floats.

    Returns
    -------
    None.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], marker='o', s=4)
    ax.set_ylim(0, 2.5)
    ax.set_title('Plot of data')
    ax.scatter(data[:,0], general_function(data[:,0], start_m_z, start_gamma_z))
    ax.scatter(data[:,0], general_function(data[:,0], m_z, gamma_z))
    plt.show()

    return None

def main():
    data = np.vstack((filter(read_data(FILE_NAME1)),filter(read_data(FILE_NAME2))))
    expected_m_z, expected_gamma_z, chi = hill_climbing(lambda coefficient1, coefficient2: chi_square(data[:,1], data[:,2], general_function(data[:,0], coefficient1, coefficient2)), start_m_z, start_gamma_z)
    print(expected_m_z)
    print(expected_gamma_z)
    print(chi)
    plot_data(data, expected_m_z, expected_gamma_z)

    return 0

if __name__ == "__main__":
    main()