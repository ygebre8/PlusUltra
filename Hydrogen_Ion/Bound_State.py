if True:
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    import h5py
    import sys
    import json
    import H2_Module as Mod

def Field_Free_Wavefunction_Reader(Target_File, input_par):
    FF_WF = {}
    psi = Target_File["Psi_" + "0" +"_" + "0"]
    psi = psi[:,0] + 1.0j*psi[:,1]
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    r_ind_max = len(grid)
    r_ind_lower = 0
    r_ind_upper = r_ind_max

    for l in range(input_par["l_max"] + 1):
        FF_WF[(l)] = np.array(psi[r_ind_lower: r_ind_upper])
 

        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max

    return(FF_WF)

if __name__=="__main__":
    file = sys.argv[1]
    
    input_par = Mod.Input_File_Reader(file + "input.json")
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    
    Target_File = h5py.File(file + "/" + input_par["Target_File"])

    FF_WF = Field_Free_Wavefunction_Reader(Target_File, input_par)

    psi = np.zeros(len(FF_WF[0]))

    for k in FF_WF.keys():
        psi += np.absolute(FF_WF[k])
        break

    plt.plot(grid, np.absolute(psi))
    # plt.xlim(0, 3)
    plt.savefig("pic_2.png")
