if True:
    import numpy as np
    import sys
    import json
    import h5py
    sys.path.append('/home/becker/yoge8051/Research/PlusUltra/TDSE')
    import Module as Mod


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

def Continuum_Wavefuncion_Reader(input_par):
    CWF_Psi = {}
    CWF_Energy = {}
    CWF_File = h5py.File("Continum.h5")

    for l in range(0, input_par["l_max"]+ 1):
        number_of_eigenvalues = int(np.array(CWF_File["ESS_" + str(l)])[0][0])

        for i in range(number_of_eigenvalues):
            dataset_name = "Energy_" + str(l) + "_" + str(i)
            CWF_Energy[l,i] = CWF_File[dataset_name]
            CWF_Energy[l,i] = np.array(CWF_Energy[l,i][:,0] + 1.0j*CWF_Energy[l,i][:,1]).real[0]

            dataset_name = "Psi_" + str(l) + "_" + str(i)
            CWF_Psi[l,i] = CWF_File[dataset_name]
            CWF_Psi[l,i] = np.array(CWF_Psi[l,i][:,0] + 1.0j*CWF_Psi[l,i][:,1])         
            norm = np.array(CWF_Psi[l,i]).max()
            CWF_Psi[l,i] /= norm

    return CWF_Energy, CWF_Psi 

