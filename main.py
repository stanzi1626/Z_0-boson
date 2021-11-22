import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.core.defchararray import splitlines

FILE_NAME1 = 'z_boson_data_1.csv'
FILE_NAME2 = 'z_boson_data_2.csv'

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
    try:
        input_file = open(filname, 'r')
        DATA_OPEN = True

    except FileNotFoundError:
        print('Unable to open file.')

    if DATA_OPEN:

        data_array = np.zeros((0, 3))
        SKIPPED_FIRST_LINE = False
        for line in input_file:

            if not SKIPPED_FIRST_LINE:
                SKIPPED_FIRST_LINE = True

            else:

                split_up = line.split(',')
                try:
                    temp = np.array([float(split_up[0]), float(split_up[1]), float(split_up[2])])
                    if np.isnan(temp[0]) or np.isnan(temp[1]) or np.isnan(temp[2]):
                        print('yikes')
                    else:    
                        data_array = np.vstack((data_array, temp))
                except ValueError:
                    pass
    input_file.close()

    return data_array

def plot_data(data1, data2):
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
    ax.scatter(data1[:, 0], data1[:, 1], marker='o', s=4)
    ax.scatter(data2[:, 0], data2[:, 1], marker='o', s=4)
    ax.set_ylim(0, 2.5)
    ax.set_title('Plot of data')
    plt.show()

    return None


def main():
    pass
    data1 = read_data(FILE_NAME1)
    data2 = read_data(FILE_NAME2)
    plot_data(data1, data2)

    return 0

if __name__ == "__main__":
    main()