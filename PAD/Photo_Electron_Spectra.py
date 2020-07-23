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
    import Potential as Pot
    import Module as Mod
    import TISE as TS
    from scipy.special import sph_harm

if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    import slepc4py 
    from slepc4py import SLEPc
    petsc4py.init(comm=PETSc.COMM_WORLD)
    slepc4py.init(sys.argv)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    

def Psi_Reader(input_par):
    TDSE_file =  h5py.File(input_par["TDSE_File"])

    Psi_Dictionary = {}
    psi  = TDSE_file["Psi_Final"]
    psi = psi[:,0] + 1.0j*psi[:,1]
    index_map_l_m, index_map_box =  eval("Mod.Index_Map_" + input_par["block_type"] + "(input_par)")
    
    r_ind_max = int(input_par["grid_size"] / input_par["grid_spacing"])
    r_ind_lower = 0
    r_ind_upper = r_ind_max
    
    for i in index_map_box:   
        Psi_Dictionary[i] = np.array(psi[r_ind_lower: r_ind_upper])
        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max
      
    return Psi_Dictionary

def Bound_Reader(input_par):
    Target_file =h5py.File(input_par["Target_File"])

    Bound_Dictionary = {}
    n_max = input_par["n_max"]
    n_values = np.arange(1, n_max + 1)
    
    for l in range(n_max):
        for n in range(l + 1, n_max + 1):
            group_name = "Psi_" + str(l) +"_" + str(n)
            Bound_Dictionary[(n, l)] = Target_file[group_name]
            Bound_Dictionary[(n, l)] = np.array(Bound_Dictionary[(n, l)][:,0] + 1.0j*Bound_Dictionary[(n, l)][:,1])
	
    return Bound_Dictionary

def Shooting_Method(k, l, input_par, z = 1 ):

    dr = input_par["grid_spacing"]
    r = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    r2 = r*r

    energy = np.power(k,2.0)/2

    potential = -1.0/r 
    coul_wave = np.zeros(r.shape)
    coul_wave[0] = 1.0
    coul_wave[1] = coul_wave[0]*(dr*dr*((l*(l+1)/r2[0])+2*potential[0]+2*energy)+2)

    for idx in np.arange(2, r.shape[0]):
        coul_wave[idx] = coul_wave[idx-1]*(dr*dr*((l*(l+1)/r2[idx-1]) + 2*potential[idx-1]-2*energy)+2) - coul_wave[idx-2]

    r_val = r[-2]
    
    coul_wave_r = coul_wave[-2]
    dcoul_wave_r = (coul_wave[-1]-coul_wave[-3])/(2*dr)

    norm = np.sqrt(np.abs(coul_wave_r)**2+(np.abs(dcoul_wave_r)/(k+z/(k*r_val)))**2)
    coul_wave /= norm

    phase =  np.angle((1.j*coul_wave_r + dcoul_wave_r/(k+z/(k*r_val))) /
                     (2*k*r_val)**(1.j*z/k)) - k*r_val + l*np.pi/2
  
   
    return phase, coul_wave[:r.shape[0]]

def K_Sphere(k, Psi, Bound_States, input_par):
    # phi = np.arange(-np.pi, np.pi, input_par["d_angle"]*100)
    # theta = np.arange(0, np.pi+input_par["d_angle"], input_par["d_angle"]*100)    
    # theta, phi = np.meshgrid(theta, phi)
    # out_going_wave = np.zeros(phi.shape, dtype=complex)

    coef_dic = {}
    for l in range(0, input_par["l_max"]+ 1):    
        phase, coul_wave = Shooting_Method(k, l, z = 1)
        n_max = input_par["n_max"]
        m_range = min(l, input_par["m_max"])
        for m in range(-1*m_range, m_range + 1):
            for n in range(l + 1, n_max + 1):
                Psi[(l, m)] -= np.sum(Bound_States[(n,l)].conj()*Psi[(l,m)])*Bound_States[(n,l)]
            # np.exp(-1.j*phase)*1.j**l *
            coef =  np.sum(coul_wave.conj()*Psi[(l,m)])
            # out_going_wave += coef*sph_harm(m, l, phi, theta)

            coef_dic[str((l,m))] = (coef.real, coef.imag)
    # print(out_going_wave, "out")
    return coef_dic


if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    Psi = Psi_Reader(input_par)
    Bound_States = Bound_Reader(input_par)
    k_max = 2
    dk = 0.01
    coef_main = {}


    for k in np.arange(dk, k_max + dk, dk): 
        k = round(k , 5)
        print(k)
        coef_dic = K_Sphere(k, Psi, Bound_States, input_par)
        coef_main[str(k)] = coef_dic

    with open("PAD.json", 'w') as file:
        json.dump(coef_main, file)

 


       
        
