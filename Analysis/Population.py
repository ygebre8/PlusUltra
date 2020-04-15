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
    sys.path.append('/home/becker/yoge8051/Research/PlusUltra/TDSE')
    import Module as Mod


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
    index_map_l_m, index_map_box =  eval("Mod.Index_Map_" + input_par["block_type"] + "(input_par)")
    

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


    index_map_l_m, index_map_box = eval("Mod.Index_Map_" + input_par["block_type"] + "(input_par)")
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
            if input_par["block_type"] == "L_Block":
                l = idx[0]
                m = idx[1]
            if input_par["block_type"] == "M_Block":
                l = idx[1]
                m = idx[0]
            
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
        max_elements = Max_Elements(N_L_Pop)
        if max_elements[1] == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(max_elements[1]))
    else:
        if vmax == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(vmax))

    label = [pow(10.0, i) for i in np.arange(vmaxlog - range, vmaxlog)]
    vmax_num = pow(10.0, vmaxlog)
    vmin_num = pow(10.0, vmaxlog - range )
    
    
    # sns.set(font_scale=1.6)
    axes = sns.heatmap(heat_map, norm=colors.LogNorm(), yticklabels=ylabel, xticklabels=xlabel, linewidths=0.01, 
    cmap="viridis", annot=False, cbar_kws={"ticks":label},  vmin=vmin_num, vmax=vmax_num,)
    

    plt.xticks(xticks, fontsize=17, rotation=90)
    plt.yticks(yticks, fontsize=17, rotation='vertical')

    plt.ylim(1, n_max + 1)
    plt.xlim(-0.5,n_max)

    plt.axvline(x=-0.5)
    plt.axhline(y=1)
    plt.xlabel('$\it{l}$',fontsize=22)
    plt.ylabel('$\it{n}$', fontsize=18)

    # if m_value == None:
    #     # plt.title("N and L States Population")
    # else:
        # plt.title("N and L States Population for m = " + str(m_value))
    
    for tick in axes.get_xticklabels():
        tick.set_rotation(360)
    
    for tick in axes.get_yticklabels():
        tick.set_rotation(360)

    print(m_value)
    if m_value == 1:
        name = "(a)"
    elif m_value == 2:
        name = "(b)"
    elif m_value == 3:
        name = "(c)"
    elif m_value == 4:
        name = "(d)"
    else:
        name = ""
    font = {'size': 22}
    matplotlib.rc('font', **font)

    plt.text(12, 5, name, fontsize=22)
    plt.savefig(file_name)
    # plt.show()
    plt.clf()

def N_L_Given_M_Population_Plotter(N_L_Pop_Given_M):
    vmax = pow(10, -20)
    for m in N_L_Pop_Given_M.keys():
        vmax_current = Max_Elements(N_L_Pop_Given_M[m])[1]
        if(vmax_current > vmax):
            vmax = vmax_current
    for m in N_L_Pop_Given_M.keys():
        if abs(m) > 6:
            continue
        if m >= 0:
            file_name = "Population_Counter-Rotating_For_M=" + str(m).zfill(2) + ".png" 
        else:
            file_name = "Population_Counter-Rotating_For_M=" + str(m).zfill(3) + ".png"

        N_L_Population_Plotter(N_L_Pop_Given_M[m], file_name, 5, m,  vmax)

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

    
    

    max_elements = Max_Elements(N_M_Pop)
    if max_elements[1] == 0:
        vmaxlog = -20
    else:
        vmaxlog = int(np.log10(max_elements[1]))

    label = [pow(10.0, i) for i in np.arange(vmaxlog - range, vmaxlog)]
    vmax_num = pow(10, vmaxlog)
    vmin_num = pow(10, vmaxlog - range)



    axes = sns.heatmap(heat_map, norm=colors.LogNorm(), yticklabels=ylabel, xticklabels=xlabel, linewidths=.1, 
    cmap="viridis", annot=False, cbar_kws={"ticks":label},  vmin=vmin_num, vmax=vmax_num)

    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.ylim(2, n_max + 1)
    # plt.xlim(np.amax(m_values) - n_max + 1, np.amax(m_values) + n_max)
    plt.xlabel('m', fontsize=18)
    plt.ylabel('n', fontsize=18)

    
    plt.title("N and M States Population", fontsize=12)
    for tick in axes.get_xticklabels():
        tick.set_rotation(360)
    
    for tick in axes.get_yticklabels():
        tick.set_rotation(360)

    font = {'size': 22}
    plt.axvline(x=0)
    plt.axhline(y=2)

    plt.savefig(file_name)
    plt.clf()

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

def L_Distribution(Population):
    excit = 0.0
    for k in Population.keys():
        if k[0] != 1:
            excit += Population[k]
        
    l_array = {}
    l_values = np.arange(1,15)

    for l in l_values:
        l_array[l] = 0.0

    for k in Population.keys():
        n = k[0]
        l = k[1]
        if l==0:
            continue
        # if n <= 3:
        #     continue

        l_array[l] += Population[k] / excit

    plt.bar(l_array.keys(), l_array.values(), align='center', alpha=1, log=True, color = 'blue')
    
    label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    plt.xlabel('$\it{l}$', fontsize=28)
    plt.ylabel("Probability", fontsize=14.2)
    plt.xticks(l_values, label, fontsize = 13)
    plt.yticks(fontsize = 14.3)
    v=list(l_array.values())
    k=list(l_array.keys())

    plt.ylim(pow(10,-6), pow(10, 0))
    plt.xlim(0.5,12)
    # plt.axvline(x=3.05, color='r')
    
    # font = {'size': 24}
    # matplotlib.rc('font', **font)

    plt.text(10, pow(10,-1), "(d)", fontsize=24)
    plt.savefig("Exc_State_Dist_1_13_and_5_13.png")
    plt.clf()

def M_Distribution(Population):
    m_array = {}
    m_values = np.arange(-13, 14, 1)
    for m in m_values:
        m_array[m] = 0.0

    for k in Population.keys():
        m = k[2]
        n = k[0]
        if n >= 4:
            m_array[m] += Population[k]
        
    
    print(m_array.keys())
    print(m_array.values())

    plt.bar(m_array.keys(), m_array.values(), align='center', alpha=1, log = True, color = 'blue')
    # plt.title("Distribution of m quantum number")
    plt.ylim(pow(10,-7), pow(10, -1))
    xticks = m_values
    xlabel = m_values
    plt.xlabel("m", fontsize=18)
    plt.ylabel("Probability", fontsize=15)
    plt.xticks(xticks, fontsize = 12)
    plt.yticks(fontsize = 14)
    
    plt.xlim(-8,8)

    # plt.savefig("M_Dist_A.png")

def Population_Comparison(Pop_1, Pop_2, Pop_3):
    l_array_1 = {}
    l_array_2 = {}
    l_array_3 = {}

    l_values = np.arange(0,15)

    for l in l_values:
        l_array_1[l] = 0.0
        l_array_2[l] = 0.0
        l_array_3[l] = 0.0

    for k in Pop_1.keys():
        n = k[0]
        l = k[1]
        # if l==0:
        #     continue
        l_array_1[l] += Pop_1[k]
        l_array_2[l] += Pop_2[k]
        l_array_3[l] += Pop_3[k]

    plt.plot(l_array_1.keys(), l_array_1.values(), 'r', label='Veloctiy 45')
    plt.plot(l_array_2.keys(), l_array_2.values(), '--k', label='Length 50')
    plt.plot(l_array_3.keys(), l_array_3.values(), '--b', label='Length 80')
    plt.legend()
    plt.yscale('log')
    plt.ylim(pow(10,-12), pow(10,0))
    plt.savefig("Gauge_Comparison.png")
    plt.clf()

def Time_Propagated_Wavefunction_Reader_Joel(TDSE_file):
    T_P_WF = {} 

    time = np.array(TDSE_file['Wavefunction']['time'])
    psi = TDSE_file['Wavefunction']['psi'][time.size - 1]
    psi = psi[:,0] + 1.0J*psi[:,1] 

    l_values = np.array(TDSE_file['Wavefunction']['l_values'])
    m_values = np.array(TDSE_file['Wavefunction']['m_values'])
    r_ind_max = TDSE_file['Wavefunction']['x_value_2'].size
    r_ind_lower = 0
    r_ind_upper = r_ind_max

    for i in range(len(l_values)):
	    T_P_WF[(l_values[i], m_values[i])] = np.array(psi[r_ind_lower: r_ind_upper])
	    r_ind_lower = r_ind_upper
	    r_ind_upper = r_ind_upper + r_ind_max

    return T_P_WF

def Field_Free_Wavefunction_Reader_Joel(Target_file):
    F_F_WF = {}
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)
    
    for i in range(n_max):
        group_name = "psi_l_" + str(i)
        for k in range(n_max - i):
            F_F_WF[(k + 1 + i, i)] = Target_file[group_name]['psi'][k]
            F_F_WF[(k + 1 + i, i)] = np.array(F_F_WF[(k + 1 + i, i)][:,0] + 1.0J*F_F_WF[(k + 1 + i, i)][:,1])
	
    return(F_F_WF)

def Population_Calculator_Joel(TDSE_file, Target_file, T_P_WF, F_F_WF):
    Population = {}

    l_values = np.array(TDSE_file['Wavefunction']['l_values'])
    m_values = np.array(TDSE_file['Wavefunction']['m_values'])
    m_max = np.amax(m_values)
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)
    
    
    for l, m  in zip(l_values, m_values):
        for n in range(l + 1, n_max + 1):
            Population[(n, l, m)] = np.vdot(F_F_WF[(n, l)], T_P_WF[(l, m)])
            Population[(n, l, m)] = np.power(np.absolute(Population[(n, l, m)]),2.0)
           
    return Population


if __name__=="__main__":

    number_of_files = int(sys.argv[1])

    Populations = []
    N_L_Populations = []
    N_M_Populations = []
    for i in range(2, 2+number_of_files):
        file = sys.argv[i]
        input_par = Mod.Input_File_Reader(file + "input.json")
        TDSE_file =  h5py.File(file + "/" + input_par["TDSE_File"])
        Target_file =h5py.File(file + "/" + input_par["Target_File"])

        FF_WF = Field_Free_Wavefunction_Reader(Target_file, input)
        print("Finished FF_WF \n")
        TP_WF = Time_Propagated_Wavefunction_Reader(TDSE_file, input_par)
        print("Finished TP_WF \n")
        Pop, N_L_Pop, N_M_Pop, N_L_Pop_Given_M = Population_Calculator(TP_WF, FF_WF, input_par)
        print("Finished Calculating Populations \n")

        Populations.append(Pop)
        

        N_L_Population_Plotter(N_L_Pop, "N_L_Population_", 5)
        N_M_Population_Plotter(N_M_Pop, "N_M_Population_", 5)

        # N_L_Given_M_Population_Plotter(N_L_Pop_Given_M)
        # L_Distribution(Pop)
        # M_Distribution(Pop)



    ion = 0

    for k in Populations[0].keys():
        ion += Populations[0][k]
        ## print(k, Populations[0][k])
  

    print((1.0 - ion)*100)

    # Joel = {}
    # for k in Populations[0].keys():
    #     Joel[str(k)] = Populations[0][k]
    #     print(str(k))
    
    # json = json.dumps(Joel)
    # f = open("Job_3.json","w")
    # f.write(json)
    # f.close()
    




