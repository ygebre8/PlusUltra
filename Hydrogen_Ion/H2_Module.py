if True:
    import numpy as np
    from math import ceil, floor
    import sys
    import h5py
    import json

def Make_Grid(grid_start, grid_end, h):
    grid = np.arange(h/2, grid_end + h, h)
    return grid

def Coulomb_Eff_Potential(grid, l):
    return -1.0*np.power(grid, -1.0) + 0.5*l*(l+1)*np.power(grid, -2.0)

def Input_File_Reader(input_file = "input.json"):
    with open(input_file) as input_file:
        input_paramters = json.load(input_file)
    return input_paramters
    

