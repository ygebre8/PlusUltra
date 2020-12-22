import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from mpl_toolkits import mplot3d
sys.path.append('/home/becker/yoge8051/Research/PlusUltra/TDSE')
import Laser_Pulse as LP 
import Module as Mod 

""" This function reads in the  dipole acc data and does the fft and gives the harmonic axis """
def Get_HHG_Data(axis, energy, file_name):
    time = np.loadtxt(file_name + "/time.txt")
    if axis == "x":
        data = np.loadtxt(file_name + "/Dip_Acc_X.txt").view(complex)
    if axis == "y":
        data = np.loadtxt(file_name + "/Dip_Acc_Y.txt").view(complex)
    if axis == "z":
        data = np.loadtxt(file_name + "/Dip_Acc_Z.txt").view(complex)

    data *= -1.0
    data = data * np.blackman(data.shape[0])

    padd2 = 2**np.ceil(np.log2(data.shape[0] * 4)) 
    paddT = np.max(time) * padd2 / data.shape[0]
    dH = 2 * np.pi / paddT / energy
    
    hhg_data = np.fft.fft(
                        np.lib.pad(
                            data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                   int(np.ceil((padd2 - data.shape[0]) / 2))),
                            'constant',
                            constant_values=(0.0, 0.0)))
    harmonic = np.arange(hhg_data.shape[0]) * dH

    return harmonic, hhg_data
   
def Make_Circ_HHG_Plot(energy, file_name):
    harmonic_x, data_x = Get_HHG_Data("x", energy, file_name)
    harmonic_y, data_y = Get_HHG_Data("y", energy, file_name)

    right = np.absolute(data_x + 1.0j*data_y)
    left = np.absolute(data_x - 1.0j*data_y)

    data_x = np.absolute(data_x)
    data_y = np.absolute(data_y)

    
    right_max = right.max()
    left_max = left.max()

    """This is normalizing the HHG spectrum """
    if right_max >= left_max:
        right /= right_max
        left /=right_max

    else:
        right /= left_max
        left /=left_max

    plt.semilogy(harmonic_x, np.power(data_x, 2.0), label = "X-Axis")
    plt.semilogy(harmonic_y, np.power(data_y, 2.0), label = "Y-Axis")
    plt.xticks(np.arange(0 + 1, 25 + 1, 2.0))
    plt.xlim(0, 25)
    plt.ylim([1e-4, 10e8])
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Helium_Cart.png")
    plt.clf()


    plt.semilogy(harmonic_x, np.power(right, 2.0), label = "Right", color = 'r')
    plt.semilogy(harmonic_x, np.power(left, 2.0), label = "Left", color = 'b')
    plt.xticks(np.arange(0 + 1, 25 + 1, 2.0))
    plt.xlim(0, 25)
    plt.ylim([1e-4, 10e8])
    plt.grid(True, which='both')#, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Helium_Circ.png")
    plt.clf()

def Linear_HHG_Plot(energy, file_name, color, shift):
    harmonic_x, data_x = Get_HHG_Data("x", energy, file_name)
    harmonic_y, data_y = Get_HHG_Data("y", energy, file_name)
    # harmonic_z, data_z = Get_HHG_Data("z", energy, file_name)

    data_x = np.absolute(data_x)
    data_y = np.absolute(data_y)
    # data_z = np.absolute(data_z)



    plt.semilogy(harmonic_x, np.power(data_x/data_x.max(), 2.0), label = "X-Axis")
    plt.semilogy(harmonic_y, np.power(data_y/data_y.max(), 2.0), label = "Y-Axis")
    # plt.semilogy(harmonic_z, np.power(data_z/data_z.max(), 2.0) * shift, label = "Z-Axis", color = color)

    plt.xticks(np.arange(0 + 1, 20 + 1, 1.0))
    plt.xlim(0, 20)
    plt.ylim([1e-12, 1])
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
   

def Full_HHG(axis, energy, file_name):
    time = np.loadtxt(file_name + "/time.txt")
    if axis == "x":
        data = np.loadtxt(file_name + "/Dip_Acc_X.txt").view(complex)
    if axis == "y":
        data = np.loadtxt(file_name + "/Dip_Acc_Y.txt").view(complex)
    if axis == "z":
        data = np.loadtxt(file_name + "/Dip_Acc_Z.txt").view(complex)

    data *= -1.0
    data = data * np.blackman(data.shape[0])
    padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
    hhg_data = np.fft.fft(data)

    T = np.max(time) / hhg_data.shape[0]
    dH = 2 * np.pi / energy
    harmonic = np.linspace(0.0, 1.0/(2.0*T), hhg_data.shape[0] / 2) * dH
    
    return harmonic, hhg_data

def HHG_Analysis(energy, file_name):
    harmonic_x, data_x = Get_HHG_Data("x", energy, file_name)
    harmonic_y, data_y = Get_HHG_Data("y", energy, file_name)

    right = data_x + 1.0j*data_y
    left = data_x - 1.0j*data_y

    low_idx = np.argwhere(harmonic_x > 4.2)[0][0]
    high_idx = np.argwhere(harmonic_x > 6.4)[0][0]

    data_x_inv = np.conj(np.flip(data_x[low_idx:high_idx]))
    data_x_inv = np.concatenate((data_x[low_idx:high_idx + 1], data_x_inv[:len(data_x_inv) - 1]))
    data_x_inv = np.fft.ifft(data_x_inv)

    data_y_inv = np.conj(np.flip(data_y[low_idx:high_idx]))
    data_y_inv = np.concatenate((data_y[low_idx:high_idx + 1], data_y_inv[:len(data_y_inv) - 1]))
    data_y_inv = np.fft.ifft(data_y_inv)

    left_inv = np.conj(np.flip(left[low_idx:high_idx]))
    left_inv = np.concatenate((left[low_idx:high_idx + 1], left_inv[:len(left_inv) - 1]))
    left_inv = np.fft.ifft(left_inv)
    
    right_inv = np.conj(np.flip(right[low_idx:high_idx]))
    right_inv = np.concatenate((right[low_idx:high_idx + 1], right_inv[:len(right_inv) - 1]))
    right_inv = np.fft.ifft(right_inv)


    # data_x_inv = data_x_inv[110:160]
    # data_y_inv = data_y_inv[110:160]
    # left_inv = left_inv[110:160]
    # right_inv = right_inv[110:160]

    time = np.linspace(0, 1, len(data_x_inv))

    fig, axs = plt.subplots(2)
    
    axs[0].plot(data_x_inv)
    axs[0].set_title('X-Axis')
    axs[1].plot(data_y_inv)
    axs[1].set_title('Y-Axis')
    # axs[1, 0].plot(right_inv)
    # axs[1, 0].set_title('Right')
    # axs[1, 1].plot(left_inv)
    # axs[1, 1].set_title('Left')

    plt.savefig("pulse.png")

def Exp():
  
    N = 600
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x)# + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = np.fft.fft(y)
    
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    

    data = np.conj(np.flip(yf[0:N//2]))
    data = np.concatenate((yf[0:N//2 + 1], data[:len(data) - 1]))
 
    y_inv = np.fft.ifft(yf)
    data_inv = np.fft.ifft(data)

    plt.plot(y_inv)
    # # plt.plot(y)
    plt.plot(data_inv)
 
    # plt.plot(np.imag(yf))
    # plt.plot(np.imag(data))
   
    plt.grid()
    plt.savefig("exam.png")

def Dipole_Plot(file_name):
    time = np.loadtxt(file_name + "/time.txt")
    data_x = np.loadtxt(file_name + "/Dip_Acc_X.txt").view(complex)
    data_y = np.loadtxt(file_name + "/Dip_Acc_Y.txt").view(complex)
    input_par = Mod.Input_File_Reader(file_name + "/input.json")
    laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse = LP.Build_Laser_Pulse(input_par)

    
    print(round(np.average(data_x), 10))
    print(round(np.average(data_y), 10))

    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


    # ax1.plot(time, data_x, label = "x")
    # ax1.plot(laser_time, -1*137.036*np.gradient(laser_pulse['x']), label = "laser_x")
    # ax1.legend()
    # ax2.plot(time, data_y, label = "y")
    # ax2.plot(laser_time, 137.036*np.gradient(laser_pulse['y']), label = "laser_y")
    # ax2.legend()

    # plt.savefig("Dipole_1068_V.png")
    # plt.show()

if __name__=="__main__":
    
    energy = 0.114

    file_name = sys.argv[1]
    Linear_HHG_Plot(energy, file_name, 'k', pow(10,8))

    # file_name = sys.argv[2]
    # Linear_HHG_Plot(energy, file_name, 'r', pow(10,4))

    # file_name = sys.argv[3]
    # Linear_HHG_Plot(energy, file_name, 'b', pow(10,0))

    plt.savefig("HHG_Cart_VXY.png")
    plt.clf()


    # Make_Circ_HHG_Plot(energy, file_name)
    # HHG_Analysis(energy, file_name)
    # Dipole_Plot(file_name)