import numpy as np
from scipy.special import sph_harm

def PAD_Momentum(coef_main, input_par):
    
    resolution = 0.05
    x_momentum = np.arange(-1, 1, resolution)
    y_momentum = np.arange(-1, 1, resolution)
    z_momentum = np.arange(-1, 1, resolution)

    pad_value = np.zeros((x_momentum.size,y_momentum.size))

    cdef int i, j, l 
    cdef double px, py, pz, phi, theta

    for i, px in enumerate(x_momentum):
        for j, py in enumerate(y_momentum):
            pad_value_temp = 0.0
            for l, pz in enumerate(z_momentum):
                k = np.sqrt(px*px + py*py + pz*pz)
                phi = np.arctan(py/px)
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
