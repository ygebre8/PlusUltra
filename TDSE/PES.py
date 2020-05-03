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
    

def PAD_Momentum(coef_main, input_par):
    
    resolution = 0.02
    x_momentum = np.arange(-1.7, 1.7 + resolution, resolution)
    y_momentum = np.arange(-1.7 , 1.7 + resolution, resolution)
    z_momentum = np.arange(-1.7 , 1.7 + 0.05, 0.05)

    pad_value = np.zeros((x_momentum.size,y_momentum.size))

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

                coef_dic = coef_main[str(closest(list(coef_main.keys()), k))]
                pad_value_temp +=  np.abs(K_Sphere(coef_dic, input_par, phi, theta))**2

            pad_value[i, j] = pad_value_temp[0][0]

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


if __name__=="__main__":
    print("Job Started")
    input_par = Mod.Input_File_Reader("input.json")
    k_max = 1.5
    dk = 0.05
    coef_main = {}

    with open("PAD.json") as file:
        coef_main = json.load(file)

    pad_value, x_momentum, y_momentum = PAD_Momentum(coef_main, input_par)
    #pad_value = np.log(pad_value)#/ pad_value.max()
    #print(pad_value.max())

    plt.imshow(pad_value, cmap='jet')#, interpolation="spline16")#, interpolation='nearest')
    
    # plt.xticks(x_momentum, x_momentum)
    # plt.yticks(y_momentum, y_momentum)
    plt.savefig("PAD_2.png")
    # coef_dic = coef_main[str(dk)]
    # phi, theta, out_going_wave_total = K_Sphere(coef_dic, input_par)
    # out_going_wave_total = np.abs(out_going_wave_total)**2

    # for k in np.arange(2*dk, k_max, dk):
    #     k = round(k , 5) 
    #     coef_dic = coef_main[str(k)]
    #     phi, theta, out_going_wave = K_Sphere(coef_dic, input_par)
    #     out_going_wave_total += np.abs(out_going_wave)**2
    
    # out_going_wave_total = out_going_wave_total/out_going_wave_total.max()

    # X, Y, Z = out_going_wave_total*np.sin(theta)*np.cos(phi), out_going_wave_total*np.sin(theta)*np.sin(phi), out_going_wave_total*np.cos(theta)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # cmap = cm.get_cmap("viridis")
    # ax.plot_surface(X, Y, Z, facecolors=cmap(out_going_wave_total))
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # plt.savefig("PAD_New_A.png")