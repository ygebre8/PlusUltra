import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys




def HHG_Plotter(axis, energy, file_name):
    time = np.loadtxt(file + "/time.txt")
    if axis == "x":
        data = np.loadtxt(file + "/Dip_Acc_X.txt").view(complex)
    if axis == "y":
        data = np.loadtxt(file + "/Dip_Acc_Y.txt").view(complex)
    if axis == "z":
        data = np.loadtxt(file + "/Dip_Acc_Z.txt").view(complex)

    data *= -1.0
    # mu = 4 * 2 * np.log(2.0) / np.pi**2
    mu = 4.0 * (np.arcsin(np.exp(-1.0 / 4.0)))**2
    shift = (1.0 + np.sqrt(1 + mu / (10)**2)) / 2.0
    energy = energy #/shift
    data = data * np.blackman(data.shape[0])
    padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
    paddT = np.max(time) * padd2 / data.shape[0]

    dH = 2 * np.pi / paddT / energy
    # data = np.absolute(
    #                     np.fft.fft(
    #                         np.lib.pad(
    #                             data, (int(np.floor((padd2 - data.shape[0]) / 2)),
    #                                 int(np.ceil((padd2 - data.shape[0]) / 2))),
    #                             'constant',
    #                             constant_values=(0.0, 0.0))))
    data = np.fft.fft(
                        np.lib.pad(
                            data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                   int(np.ceil((padd2 - data.shape[0]) / 2))),
                            'constant',
                            constant_values=(0.0, 0.0)))
    # data /= data.max()
    # data = np.power(data, 2.0)
    
    
    if axis == "x":
        np.savetxt(file_name + "_X_.txt", data)
        np.savetxt(file_name +  "_Harmonic.txt", np.arange(data.shape[0]) * dH)
    if axis == "y":
        np.savetxt(file_name + "_Y_.txt", data)

    # plt.semilogy(np.arange(data.shape[0]) * dH, data, label = axis + "-axis")


if __name__=="__main__":
    file = sys.argv[1]
    # energy = 0.0855 
    energy = 0.057
    # energy = 0.04275
    HHG_Plotter("x", energy, "HHG_Neg_0")
    HHG_Plotter("y", energy, "HHG_Neg_0")

 

