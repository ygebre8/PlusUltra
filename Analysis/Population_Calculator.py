if True:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import h5py
    import sys
    import json
    import os
    import seaborn as sns
    sys.path.append('~/Research/PlusUltra/TDSE')
    import Module as Mod
    from numpy import *
    from matplotlib.pyplot import *

def Field_Free_Wavefunction_Reader(Target_file, input):
    FF_WF = {}
    n_max = input_par["n_max"]
    n_values = np.arange(1, n_max + 1)
    
    for l in range(n_max):
        for n in range(l + 1, n_max + 1):
            group_name = "Psi_" + str(l) +"_" + str(n)
            FF_WF[(n, l)] = Target_file[group_name]
            FF_WF[(n, l)] = np.array(FF_WF[(n, l)][:,0] + 1.0j*FF_WF[(n, l)][:,1])
	
    return(FF_WF)

def Time_Propagated_Wavefunction_Reader(TDSE_file, input_par):
    TP_WF = {}
    psi  = TDSE_file["Psi_Final"]
    psi = psi[:,0] + 1.0j*psi[:,1]
    index_map_l_m, index_map_box =  eval("Mod.Index_Map(input_par)")
    
    r_ind_max = int(input_par["grid_size"] / input_par["grid_spacing"])
    r_ind_lower = 0
    r_ind_upper = r_ind_max
    
    for i in index_map_box:   
        TP_WF[i] = np.array(psi[r_ind_lower: r_ind_upper])
        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max
      
    return TP_WF

def Population_Calculator(TP_WF, FF_WF, input_par):
    Population = {}
    N_L_Pop = {}
    N_M_Pop = {}
    N_L_Pop_Given_M = {}


    index_map_l_m, index_map_box = eval("Mod.Index_Map(input_par)")
    n_max = input_par["n_max"]
    n_values = np.arange(1, n_max + 1)
  
    for n in n_values:
        for l in range(0, n):
            N_L_Pop[(n, l)] = pow(10, -20)
        for m in np.arange(-1*n + 1, n):
            N_M_Pop[(n, m)] = pow(10, -20)

    for m in np.arange(-1*n + 1, n):
        N_L_Pop_Given_M[m] = {}
        for l in range(n_max):
            for n in range(l + 1, n_max + 1):
                N_L_Pop_Given_M[m][(n, l)] = pow(10, -20)

    for idx in index_map_box:
        for n in range(idx[0] + 1, n_max + 1):
            l = idx[0]
            m = idx[1]
            
            
            Population[(n, l, m)] = np.vdot(FF_WF[(n, l)], TP_WF[idx])
            Population[(n, l, m)] = np.power(np.absolute(Population[(n, l, m)]),2.0)

            N_L_Pop[(n, l)] = N_L_Pop[(n, l)] + Population[(n, l, m)]
            N_M_Pop[(n, m)] = N_M_Pop[(n, m)] + Population[(n, l, m)]

            N_L_Pop_Given_M[m][(n, l)] = N_L_Pop_Given_M[m][(n, l)] + Population[(n, l, m)]
        

    return Population, N_L_Pop, N_M_Pop, N_L_Pop_Given_M





if __name__=="__main__":
    file = sys.argv[1]
    input_par = Mod.Input_File_Reader(file + "input.json")
    TDSE_file =  h5py.File(file + "/" + input_par["TDSE_File"])
    Target_file =h5py.File(file + "/" + input_par["Target_File"])

    FF_WF = Field_Free_Wavefunction_Reader(Target_file, input)
    print("Finished FF_WF \n")
    TP_WF = Time_Propagated_Wavefunction_Reader(TDSE_file, input_par)
    print("Finished TP_WF \n")
    Pop, N_L_Pop, N_M_Pop, N_L_Pop_Given_M = Population_Calculator(TP_WF, FF_WF, input_par)
    print("Finished Calculating Populations \n")

    bound = 0
    for k in Populations[0].keys():
        bound += Populations[0][k]

    print((1.0 - bound)*100)