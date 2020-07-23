import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def HHG_Plotter(file):


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.clf()
    Harmonic = np.loadtxt(file + "HHG_Neg_" + str(0) + '_Harmonic.txt', dtype=float)
    data_x = np.loadtxt(file + "HHG_Neg_" + str(0) + '_X_.txt', dtype=complex)
    data_y = np.loadtxt(file + "HHG_Neg_" + str(0) + '_Y_.txt', dtype=complex)

    right = np.absolute(data_x + 1.0j*data_y)
    left = np.absolute(data_x - 1.0j*data_y)
    data_x = np.absolute(data_x)
    data_y = np.absolute(data_y)

    # data_x = np.power(data_x/data_x.max(), 2.0)
    # data_y = np.power(data_y/data_y.max(), 2.0)

    data_x = np.power(data_x, 2.0)
    data_y = np.power(data_y, 2.0)

    right_max = right.max()
    left_max = left.max()

    
    # if right_max >= left_max:
    #     print("here")
    #     right /= right_max
    #     left /=right_max

    # if right_max < left_max:
    #     print("2here")
    #     right /= left_max
    #     left /=left_max

    right = np.power(right, 2.0)
    left = np.power(left, 2.0)


    plt.semilogy(Harmonic, data_x, label = "X-Axis")
    plt.semilogy(Harmonic, data_y, label = "Y-Axis")
    plt.xticks(np.arange(0 + 1, 25 + 1, 1.0))
    plt.xlim(0, 20)
    plt.ylim([1e-12, 1])
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig("HHG_Cart_Cou_534.png")
    plt.clf()

    # Prediced_Lines(4)
    plt.semilogy(Harmonic, right, label = "Right", color = 'r')
    plt.semilogy(Harmonic, left, label = "Left", color = 'b')
    plt.xticks(np.arange(0 + 1, 25 + 1, 1.0))
    plt.xlim(0, 20)
    plt.ylim([1e-12, 1])
    plt.grid(True, which='both')#, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig("HHG_Circ_Cou_534.png")
    plt.clf()

def Prediced_Lines(n):
    p_array = np.arange(10)
    for p in p_array:
        plt.axvline(x = (n-1)*p + (2-n), color = 'b')
        if p == 9:
            continue
        plt.axvline(x = (n-1)*p + (n-2) + 0.06, color = 'r')
if __name__=="__main__":
    
    # file = '/home/becker/yoge8051/Research/Jobs/Lamda/Diff_Intensity/HHG_Data/'
    # file = '/home/becker/yoge8051/Research/Jobs/Lamda/Eight_Hundred_NM/5_13_and_1_14/'
    # file = '/home/becker/yoge8051/Research/Bryn/800_80_Per/'
    # file = '/home/becker/yoge8051/Research/Jobs/Lamda/One_Sixty_Eight_NM/5_13_and_1_14/'
    file = '/mpdata/becker/yoge8051/Counter/Job_1/'


    HHG_Plotter(file)