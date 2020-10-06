if True:
    import numpy as np
    from math import ceil
    import sys
    import h5py
    import json

def Make_Grid(grid_start, grid_end, h):
    number_of_grid_points = ceil((grid_end - grid_start) / h) + 1
    grid = np.linspace(grid_start, grid_end, number_of_grid_points)
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


def Max_Elements(input_dict):
    max_element = 0.0
    max_element_second = 0.0

    for k in input_dict.keys():
        if(input_dict[k] > max_element):
            max_element = input_dict[k]
    for k in input_dict.keys():
        if(input_dict[k] > max_element_second and input_dict[k] < max_element):
            max_element_second = input_dict[k]
    return (max_element, max_element_second)