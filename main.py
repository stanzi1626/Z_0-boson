import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize
from mpl_toolkits import mplot3d

FILE_NAME1 = 'z_boson_data_1.csv'
FILE_NAME2 = 'z_boson_data_2.csv'

gamma_ee = 0.08391 #GeV
start_gamma_z = 3 #Gev
start_m_z = 90 #Gev/c^2 #values should be around these
uncertainty_confidence = 3

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


def reduced_chi_square(observation, observation_uncertainty, prediction):
    """
    Returns the reduced chi sqaured
    """
    return (np.sum(((observation - prediction) / observation_uncertainty)**2)) / len(observation - 1)

def find_parameters(data):
    """
    
    """
    x, y = scipy.optimize.curve_fit(general_function, data[:,0], data[:,1], sigma = data[:, 2], p0=[start_m_z, start_gamma_z])
    return x[0], x[1]

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

def uncertainty_filter(data, m_z, gamma_z):
    """
    
    """
    count = 0
    index = 0
    for line in data:
        if line[1] + line[2]*uncertainty_confidence < general_function(line[0], m_z, gamma_z) or line[1] - line[2]*uncertainty_confidence > general_function(line[0], m_z, gamma_z):
            data = np.delete(data, index, axis=0)
            count += 1
            index -= 1
        index += 1
    return data, count

def find_final_parameters(data):
    """
    
    """
    while True:
        m_z, gamma_z = find_parameters(data)
        data, count = uncertainty_filter(data, m_z, gamma_z)
        if count == 0:
            break
    return data, m_z, gamma_z

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

    m_data = np.linspace(m_z - 10, m_z + 10,len(data[:,0]))
    gamma_data = np.linspace(gamma_z - 1, gamma_z + 1, len(data[:,0]))

    chi_m_data = []
    for i in range(len(data[:,0])):
        chi_m_data.append(reduced_chi_square(data[:,1], data[:,2], general_function(data[:,0], m_data[i], gamma_z)))
    
    chi_gamma_data = []
    for i in range(len(data[:,0])):
        chi_gamma_data.append(reduced_chi_square(data[:,1], data[:,2], general_function(data[:,0], m_z, gamma_data[i])))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='o')
    ax1.set_title('Cross-sectional area against Energy')
    ax1.set_xlabel('Energy / Gev')
    ax1.set_ylabel('Cross-sectional area / nb')
    '''ax1.scatter(data[:,0], general_function(data[:,0], start_m_z, start_gamma_z))'''
    ax1.scatter(data[:,0], general_function(data[:,0], m_z, gamma_z))

    ax2.set_title('Reduced chi-sqaured against varying m_z')
    ax2.set_xlabel('m_z / Gev*c^-2')
    ax2.set_ylabel('Reduced chi-sqaured')
    ax2.scatter(m_data, chi_m_data)

    ax3.set_title('Reduced chi-sqaured against varying gamma_z')
    ax3.set_xlabel('gamma_z / Gev')
    ax3.set_ylabel('Reduced chi-sqaured')
    ax3.scatter(gamma_data, chi_gamma_data)
    plt.show()

    return None

def plot_3d(data, true_m_z, true_gamma_z):
    """
    
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    diference1 = 0.05
    difference2 = 0.05

    x = np.linspace(true_m_z + diference1, true_m_z - diference1, len(data[:,0]))
    y = np.linspace(true_gamma_z + difference2, true_gamma_z - difference2, len(data[:,0]))
    
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((0, len(X[:,0])))
    index = 0
    for line in Y:
        temp = []
        for i in range(len(X[:,0])):
            temp.append(reduced_chi_square(data[:,1], data[:,2], general_function(data[:,0], X[index, i], line[i])))
        Z = np.vstack((Z, temp))
        index += 1

    ax.scatter3D(X, Y, Z)
    ax.set_xlim3d(true_m_z + diference1, true_m_z - diference1)
    ax.set_ylim3d(true_gamma_z + difference2, true_gamma_z - difference2)
    ax.set_zlim3d(np.min(Z), np.max(Z))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def main():
    """
    
    """
    data = np.vstack((filter_initial(read_data(FILE_NAME1)),filter_initial(read_data(FILE_NAME2))))
    data, expected_m_z, expected_gamma_z = find_final_parameters(data)
    plot_data(data, expected_m_z, expected_gamma_z)
    plot_3d(data, expected_m_z, expected_gamma_z)

    return 0

if __name__ == "__main__":
    main()