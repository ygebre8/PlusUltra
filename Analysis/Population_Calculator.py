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
    import warnings
    import Module as Mod


def Field_Free_Wavefunction_Reader(Target_file, input):
    FF_WF = {}
    n_max = input_par["n_max"]
    n_values = np.arange(1, n_max + 1)
    
    for l in range(n_max):
        for n in range(l + 1, n_max + 1):
            group_name = "BS_Psi_" + str(l) +"_" + str(n)
            FF_WF[(n, l)] = Target_file[group_name]
            FF_WF[(n, l)] = np.array(FF_WF[(n, l)][:,0] + 1.0j*FF_WF[(n, l)][:,1])
	
    return(FF_WF)

def Time_Propagated_Wavefunction_Reader(TDSE_file, input_par):
    TP_WF = {}
    psi  = TDSE_file["Psi_Final"]
    psi = psi[:,0] + 1.0j*psi[:,1]
    index_map, index_map_box =  eval("Mod.Index_Map(input_par)")
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    
    r_ind_max = len(grid)
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

def N_L_Population_Plotter(N_L_Pop, file_name, range, m_value = None, vmax = None):
    
    n_max = input_par["n_max"]
    n_values = np.arange(1, n_max + 1)
    l_max = n_max - 1
    l_values = np.arange(0, n_max)
    heat_map = np.zeros(shape=(n_max + 1,n_max))
    
    for n in n_values:
        for l  in np.arange(0, n):
            heat_map[n][l] = N_L_Pop[(n, l)]
            if m_value != None:
                if l < abs(m_value):
                    heat_map[n][l] = None
            
    heat_map[0][:] = None
    
    figure = plt.figure()
    axes = figure.add_subplot(1,1,1)
    xticks = np.arange(0.5, n_max, 2)
    yticks = np.arange(1.5, n_max + 1, 2)
    xlabel = np.arange(0, n_max, 2)
    ylabel = np.arange(1, n_max + 1, 2)
    
    if vmax == None:
        max_elements = Mod.Max_Elements(N_L_Pop)
        if max_elements[1] == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(max_elements[1]))
    else:
        if vmax == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(vmax))

    label = [pow(10.0, i) for i in np.arange(vmaxlog - range,  vmaxlog)]
    vmax_num = pow(10.0, vmaxlog  - 0.1)
    vmin_num = pow(10.0, vmaxlog - range)
    
    sns.set(font_scale=1.75)
    axes = sns.heatmap(heat_map, norm=colors.LogNorm(), yticklabels=ylabel, xticklabels=xlabel, linewidths=0.01, 
    cmap="viridis", annot=False, cbar_kws={"ticks":label},  vmin=vmin_num, vmax=vmax_num,)

    plt.xticks(xticks, fontsize=17, rotation=90, fontweight='bold')
    plt.yticks(yticks, fontsize=17, rotation='vertical', fontweight='bold')

    plt.ylim(2, n_max + 1)
    plt.xlim(-0.5,n_max)

    plt.axvline(x=-0.5)
    plt.axhline(y=1)

    plt.ylabel( r"$\mathbf{n}$", fontsize=32, fontweight='bold')
    plt.xlabel( r"$\mathbf{l}$", fontsize=32, fontweight='bold')

    # plt.text(7, -1, r"$\mathbf{l}$", fontsize=24, fontweight='bold', color = 'k')

    if m_value == None:
        plt.title("N and L States Population")
    else:
        plt.title("N and L States Population for m = " + str(m_value))
    
    for tick in axes.get_xticklabels():
        tick.set_rotation(360)
    
    for tick in axes.get_yticklabels():
        tick.set_rotation(360)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()

def N_M_Population_Plotter(N_M_Pop, file_name, range):
    
    n_max = input_par["n_max"]
    n_values = np.arange(1, n_max + 1)
    l_max = n_max - 1
    m_max = 2 * l_max + 1
    l_values = np.arange(0, n_max)
    m_values = l_values
    heat_map = np.zeros(shape=(n_max + 1, m_max))


    for n in n_values:
        for m in np.arange(-1*n + 1, n):
            heat_map[n][m + np.amax(m_values)] = N_M_Pop[(n, m)]
            

    figure = plt.figure()
    axes = figure.add_subplot(1,1,1)
    xticks = np.arange(1.5, 2*np.amax(m_values) + 1, 2)
    yticks = np.arange(1.5, n_max + 1,2)
    ylabel = np.arange(1, n_max + 1, 2)
    xlabel = np.arange(-1 * np.amax(m_values) + 1, np.amax(m_values) + 7, 2)


    max_elements = Mod.Max_Elements(N_M_Pop)
    if max_elements[1] == 0:
        vmaxlog = -20
    else:
        vmaxlog = int(np.log10(max_elements[1]))

    label = [pow(10.0, i) for i in np.arange(vmaxlog - range, vmaxlog)]
    vmax_num = pow(10, vmaxlog)
    vmin_num = pow(10, vmaxlog - range)

    sns.set(font_scale=1.75)
    axes = sns.heatmap(heat_map, norm=colors.LogNorm(), yticklabels=ylabel, xticklabels=xlabel, linewidths=.1, 
    cmap="viridis", annot=False, cbar_kws={"ticks":label},  vmin=vmin_num, vmax=vmax_num)

    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.ylim(2, n_max + 1)
    # plt.xlim(np.amax(m_values) - n_max + 1, np.amax(m_values) + n_max)
   
   
    plt.ylabel( r"$\mathbf{n}$", fontsize=24, fontweight='bold')
    plt.xlabel( r"$\mathbf{m}$", fontsize=24, fontweight='bold')

    plt.title("N and M States Population", fontsize=12)

    for tick in axes.get_xticklabels():
        tick.set_rotation(360)
    
    for tick in axes.get_yticklabels():
        tick.set_rotation(360)

    plt.axvline(x=0)
    plt.axhline(y=2)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()

def N_L_Given_M_Population_Plotter(N_L_Pop_Given_M, m_range):
    vmax = pow(10, -20)
    for m in N_L_Pop_Given_M.keys():
        vmax_current = Max_Elements(N_L_Pop_Given_M[m])[1]
        if(vmax_current > vmax):
            vmax = vmax_current
    for m in N_L_Pop_Given_M.keys():
        if abs(m) > m_range:
            continue
        if m >= 0:
            file_name = "plots/N_L_Pop_For_M=" + str(m).zfill(2) + ".png" 
        else:
            file_name = "plots/N_L_Pop_For_M=" + str(m).zfill(3) + ".png"

        N_L_Population_Plotter(N_L_Pop_Given_M[m], file_name, 5, m,  vmax)

if __name__=="__main__":
    
    warnings.filterwarnings("ignore",category =RuntimeWarning)
    if not os.path.exists('plots'):
        os.makedirs('plots')


    input_par = Mod.Input_File_Reader("input.json")
    TDSE_file =  h5py.File(input_par["TDSE_File"])
    Target_file =h5py.File(input_par["Target_File"])

    FF_WF = Field_Free_Wavefunction_Reader(Target_file, input)
    # print("Finished FF_WF \n")
    TP_WF = Time_Propagated_Wavefunction_Reader(TDSE_file, input_par)
    # print("Finished TP_WF \n")
    Pop, N_L_Pop, N_M_Pop, N_L_Pop_Given_M = Population_Calculator(TP_WF, FF_WF, input_par)
    # print("Finished Calculating Populations \n")
     
    
    # N_L_Population_Plotter(N_L_Pop, "plots/N_L_Pop", 5)
    # N_M_Population_Plotter(N_M_Pop, "plots/N_M_Pop", 5)
    # N_L_Given_M_Population_Plotter(N_L_Pop_Given_M, 5)

    bound = 0
    for k in Pop.keys():
        bound += Pop[k]

    print(1.0 - bound)
    # print("Ionzied percentage is " + str((1.0 - bound)*100))