from sympy.physics.wigner import gaunt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

X = True
Y = True
Z = False

file_X = "/home/becker/yoge8051/Research/Jobs/Gauge/Velocity/C_Pol"
file_Y = "/home/becker/yoge8051/Research/Jobs/Gauge/Length/C_Pol"
if X == True:

    data = np.loadtxt(file_X + "/Dip_Acc_Y.txt").view(complex)
    time = np.loadtxt(file_X + "/time.txt")


    data *= -1.0

    # mu = 4 * 2 * np.log(2.0) / np.pi**2

    mu = 4.0 * (np.arcsin(np.exp(-1.0 / 4.0)))**2

    shift = (1.0 + np.sqrt(1 + mu / (10)**2)) / 2.0

    energy = 0.375 #/ shift

    data = data * np.blackman(data.shape[0])
    padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
    paddT = np.max(time) * padd2 / data.shape[0]

    dH = 2 * np.pi / paddT / energy

    data = np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                    int(np.ceil((padd2 - data.shape[0]) / 2))),
                                'constant',
                                constant_values=(0.0, 0.0))))
    data /= data.max()
    data = np.power(data, 2.0)
    plt.semilogy(np.arange(data.shape[0]) * dH, data)

if Y == True:

    data = np.loadtxt(file_Y + "/Dip_Acc_Y.txt").view(complex)
    time = np.loadtxt(file_Y + "/time.txt")


    data *= -1.0

    # mu = 4 * 2 * np.log(2.0) / np.pi**2

    mu = 4.0 * (np.arcsin(np.exp(-1.0 / 4.0)))**2

    shift = (1.0 + np.sqrt(1 + mu / (10)**2)) / 2.0

    energy = 0.375 #/ shift

    data = data * np.blackman(data.shape[0])
    padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
    paddT = np.max(time) * padd2 / data.shape[0]

    dH = 2 * np.pi / paddT / energy

    data = np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                    int(np.ceil((padd2 - data.shape[0]) / 2))),
                                'constant',
                                constant_values=(0.0, 0.0))))
    data /= data.max()
    data = np.power(data, 2.0)
    plt.semilogy(np.arange(data.shape[0]) * dH, data)

if Z == True:

    data = np.loadtxt(file + "/Dip_Acc_Z.txt").view(complex)
    time = np.loadtxt(file + "/time.txt")


    data *= -1.0

    # mu = 4 * 2 * np.log(2.0) / np.pi**2

    mu = 4.0 * (np.arcsin(np.exp(-1.0 / 4.0)))**2

    shift = (1.0 + np.sqrt(1 + mu / (10)**2)) / 2.0

    energy = 0.375 #/ shift

    data = data * np.blackman(data.shape[0])
    padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
    paddT = np.max(time) * padd2 / data.shape[0]

    dH = 2 * np.pi / paddT / energy

    data = np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                    int(np.ceil((padd2 - data.shape[0]) / 2))),
                                'constant',
                                constant_values=(0.0, 0.0))))
    data /= data.max()
    data = np.power(data, 2.0)
    plt.semilogy(np.arange(data.shape[0]) * dH, data)


plt.xlim(0, 25)
plt.ylim([1e-16, 1])
plt.xticks(np.arange(0 + 1, 25 + 1, 2.0))
plt.grid(True, which='both')
plt.tight_layout()

plt.savefig("HHG_Y.png")