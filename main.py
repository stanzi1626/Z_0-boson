import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize

FILE_NAME1 = 'z_boson_data_1.csv'
FILE_NAME2 = 'z_boson_data_2.csv'

gamma_ee = 0.08391 #GeV
start_gamma_z = 3 #Gev
start_m_z = 90 #Gev/c^2 #values should be around these
'''STEP_SIZE = 0.0001
TOLERANCE = 0.0001
MAX_ITERATIONS = 100000'''

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
    
    """
    index = 0
    for line in data:
        for i in range(0, len(line)):
            if np.isnan(line[i]):
                data = np.delete(data, index, axis=0)
                index -= 1
                break
        index += 1
    return data


def plot_data(data):
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
    plt.show()

    return None

def main():
    data = np.vstack((filter(read_data(FILE_NAME1)),filter(read_data(FILE_NAME2))))
    expected_m_z, expected_gamma_z = find_parameters(data)
    print(expected_gamma_z)
    print(expected_m_z)
    '''plot_data(data)'''

    return 0

if __name__ == "__main__":
    main()