if True:
    import numpy as np
    from math import ceil
    import sys
    import h5py
    import json

def Make_Grid(grid_start, grid_end, h):
    grid = np.arange(h, grid_end + h, h)
    return grid

def Input_File_Reader(input_file = "input.json"):
    with open(input_file) as input_file:
        input_paramters = json.load(input_file)
    return input_paramters
    
def Index_Map(input_par):
    l_max = input_par["l_max"]
    m_max = input_par["m_max"]
    index_map_l_m = {}
    index_map_box = {}
    count = 0
    for l in np.arange(l_max + 1):
        if m_max > 0:
            for m in np.arange(-1 * l, l + 1):
                index_map_l_m[count] = (l,m)
                index_map_box[(l,m)] = count
                count += 1
        if m_max == 0:
            index_map_l_m[count] = (l,0)
            index_map_box[(l,0)] = count
            count += 1
    return index_map_l_m, index_map_box
    
def Target_File_Reader(input_par):
    file = h5py.File(input_par["Target_File"], 'r')
    energy = {}
    wave_function = {}
    for l in range(input_par["n_max"]):
        for n in range(l + 1, input_par["n_max"]+1):
            energy[(n, l)] = file["BS_Energy_" + str(l) + "_" + str(n)]
            energy[(n, l)] = np.array(energy[(n, l)][:,0] + 1.0j*energy[(n, l)][:,1])
            wave_function[(n, l)] = file["BS_Psi_" + str(l) + "_" + str(n)]
            wave_function[(n, l)] = np.array(wave_function[(n, l)][:,0] + 1.0j*wave_function[(n, l)][:,1])

    return energy, wave_function

def Matrix_Build_Status(input):

    build_status = {}

    build_status["Int_Mat_X_Stat"] = False 
    build_status["Int_Mat_Y_Stat"] = False 
    build_status["Int_Mat_Z_Stat"] = False 

    build_status["Int_Mat_Right_Stat"] = False
    build_status["Int_Mat_Left_Stat"] = False

    build_status["Dip_Acc_Mat_X_Stat"] = False
    build_status["Dip_Acc_Mat_Y_Stat"] = False
    build_status["Dip_Acc_Mat_Z_Stat"] = False
    
    build_status["Dip_Mat_X_Stat"] = False
    build_status["Dip_Mat_Y_Stat"] = False
    build_status["Dip_Mat_Z_Stat"] = False

    build_status["Int_Ham_Temp"] = False
    build_status["Int_Ham_Left_Temp"] = False

    return build_status