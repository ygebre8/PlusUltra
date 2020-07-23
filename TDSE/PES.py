if True:
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import sys
    import json
    import h5py
    import os
    import Module as Mod
    from scipy.special import sph_harm
    from matplotlib.colors import LogNorm
    from numpy import pi
    

def PAD_Momentum(COEF, input_par):
    # print(COEF.keys())
    resolution = 0.01
    x_momentum = np.arange(-1.5 ,1.5 + resolution, resolution)
    y_momentum = np.arange(-1.5 ,1.5 + resolution, resolution)

    resolution = 0.05
    z_momentum = np.arange(-1.5 ,1.5 + resolution, resolution)
    
    

    pad_value = np.zeros((y_momentum.size,x_momentum.size))

    for i, px in enumerate(x_momentum):
        print(px)
        for j, py in enumerate(y_momentum):
            pad_value_temp = 0.0
            for l, pz in enumerate(z_momentum):
            
                k = np.sqrt(px*px + py*py + pz*pz)
                if k == 0:
                    continue
                
                if px > 0 and py > 0:
                    phi = np.arctan(py/px)
                elif px > 0 and py < 0:
                    phi = np.arctan(py/px) + 2*pi
                elif px < 0 and py > 0:
                    phi = np.arctan(py/px) + pi
                elif px < 0 and py < 0:
                    phi = np.arctan(py/px) + pi
                elif px == 0 and py == 0:
                    phi = 0
                elif px == 0 and py > 0:
                    phi = pi / 2
                elif px == 0 and py < 0:
                    phi = 3*pi / 2
                elif py == 0 and px > 0:
                    phi = 0
                elif py == 0 and px < 0:
                    phi = pi

                theta = np.arccos(pz/k)
                coef_dic = COEF[closest(list(COEF.keys()), k)]
                pad_value_temp +=  np.abs(K_Sphere(coef_dic, input_par, phi, theta))**2

            pad_value[j, i] = pad_value_temp[0][0]

    return pad_value, x_momentum, y_momentum 

def K_Sphere(coef_dic, input_par, phi, theta):
    theta, phi = np.meshgrid(theta, phi)
    out_going_wave = np.zeros(phi.shape, dtype=complex)
    for l in range(0, input_par["l_max"]+ 1):    
        m_range = min(l, input_par["m_max"])
        for m in range(-1*m_range, m_range + 1):
            coef = coef_dic[str((l,m))][0] + 1j*coef_dic[str((l,m))][1]
            out_going_wave += coef*sph_harm(m, l, phi, theta)

    return out_going_wave

def closest(lst, k): 
    return lst[min(range(len(lst)), key = lambda i: abs(float(lst[i])-k))] 

def Coef_Plotter(coef_main, COEF, l, m):
    k_max = 2
    dk = 0.01
    k_array = np.arange(dk, k_max + dk, dk)
    coef = np.zeros(len(k_array), dtype=complex)
    
    for i, k in enumerate(k_array): 
        k = round(k , 5)
        coef[i] = np.absolute(coef_main[str(k)][str((l,m))][0] + 1.0j*coef_main[str(k)][str((l,m))][1])

    plt.plot(k_array, coef)

    plt.xlim(-0.1,2)
    plt.savefig("COEF_Old.png")

if __name__=="__main__":
    print("Job Started")
    input_par = Mod.Input_File_Reader("input.json")
    coef_main = {}

    with open("PAD.json") as file:
        coef_main = json.load(file)

    # Coef_Plotter(coef_main, COEF, 5, 0)

    pad_value, x_momentum, y_momentum = PAD_Momentum(coef_main, input_par)
    pad_value = pad_value / pad_value.max()
    plt.imshow(pad_value, cmap='jet')#, interpolation="spline16")#, interpolation='nearest')
    plt.savefig("PAD_Old.png")

