def Eigen_Value_Solver(Hamiltonian, number_of_eigenvalues, input_par, l, Viewer):
    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    EV_Solver.setOperators(Hamiltonian) ##pass the hamiltonian to the 
    # EV_Solver.setType(SLEPc.EPS.Type.JD)
    EV_Solver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    EV_Solver.setTolerances(input_par["tolerance"], PETSc.DECIDE)
    EV_Solver.setWhichEigenpairs(EV_Solver.Which.SMALLEST_REAL)
    size_of_matrix = PETSc.Mat.getSize(Hamiltonian)
    dimension_size = int(size_of_matrix[0]) * 0.1
    EV_Solver.setDimensions(number_of_eigenvalues, PETSc.DECIDE, dimension_size) 
    EV_Solver.solve() ##solve the eigenvalue problem
    eigen_states = [[],[]]

    if rank == 0:
        print("Number of eigenvalues requested and converged")
        print(number_of_eigenvalues, EV_Solver.getConverged(), "\n")
   
    num_of_pos_eigenvalues = 0

    for i in range(number_of_eigenvalues):
        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)

        if eigen_state.real < 0:
            continue

        eigen_vector.setName("Psi_" + str(l) + "_" + str(num_of_pos_eigenvalues)) 
        Viewer.view(eigen_vector)
        
        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        energy.setValue(0,eigen_state)
        
        energy.setName("Energy_" + str(l) + "_" + str(num_of_pos_eigenvalues))

        energy.assemblyBegin()
        energy.assemblyEnd()
        Viewer.view(energy)
        
        num_of_pos_eigenvalues+= 1

    eigen_state_solved = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
    eigen_state_solved.setValue(0,num_of_pos_eigenvalues)
    eigen_state_solved.setName("ESS_" + str(l))
    eigen_state_solved.assemblyBegin()
    eigen_state_solved.assemblyEnd()
    Viewer.view(eigen_state_solved)

def Continuum_Wavefuncion_Maker(input_par, number_of_eigenvalues):

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5("Continum.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    for l in range(0, input_par["l_max"]+ 1):

        if rank == 0:
            print("Calculating the eigenstates for l = " + str(l) + "\n")
    
        potential = eval("Pot." + input_par["potential"] + "(grid, l)")
        Hamiltonian = eval("TS.Build_Hamiltonian_" + input_par["order"] + "_Order_CW(potential, grid, input_par)")        
        Eigen_Value_Solver(Hamiltonian, number_of_eigenvalues, input_par, l, ViewHDF5)

        if rank == 0:
            print("Finished calculation for l = " + str(l) + "\n" , "\n")

    ViewHDF5.destroy()

def Continuum_Wavefuncion_Reader(input_par):
    CWF_Psi = {}
    CWF_Energy = {}
    Phase = {}
    CWF_File = h5py.File("Continum.h5")
    smallest_eigen_number = 10e10
    dr = input_par["grid_spacing"]
    r = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    r_val = r[-2]
    z = 1

    for l in range(0, input_par["l_max"]+ 1):
        number_of_eigenvalues = int(np.array(CWF_File["ESS_" + str(l)])[0][0])
        if number_of_eigenvalues < smallest_eigen_number:
            smallest_eigen_number = number_of_eigenvalues
        for i in range(number_of_eigenvalues):
            dataset_name = "Energy_" + str(l) + "_" + str(i)
            CWF_Energy[l,i] = CWF_File[dataset_name]
            CWF_Energy[l,i] = np.array(CWF_Energy[l,i][:,0] + 1.0j*CWF_Energy[l,i][:,1]).real[0]

            dataset_name = "Psi_" + str(l) + "_" + str(i)
            CWF_Psi[l,i] = CWF_File[dataset_name]
            CWF_Psi[l,i] = np.array(CWF_Psi[l,i][:,0] + 1.0j*CWF_Psi[l,i][:,1])
            
            coul_wave_r = CWF_Psi[l,i][-2]
            dcoul_wave_r = (CWF_Psi[l,i][-1]-CWF_Psi[l,i][-3])/(2*dr)

            norm = np.array(CWF_Psi[l,i]).max()
            k = np.sqrt(2.0*CWF_Energy[l,i])

            phase, coul_wave = PES.Shooting_Method(k, l, input_par, z = 1)

            CWF_Psi[l,i] /= norm
            
            CWF_Psi[l,i] = coul_wave
            


            #

            # Phase[l,i] = np.angle((1.j*coul_wave_r + dcoul_wave_r/(k+z/(k*r_val))) /
            #             (2*k*r_val)**(1.j*z/k)) - k*r_val + l*np.pi/2

            Phase[l,i] = phase

    return CWF_Energy, CWF_Psi, Phase , smallest_eigen_number 

def Coef_Calculator(input_par, Psi, Bound_States, CWF_Energy, CWF_Psi, Phase, smallest_eigen_number):

    COEF = {}
    for l in range(0, input_par["l_max"]+ 1):    
        n_max = input_par["n_max"]
        m_range = min(l, input_par["m_max"])
        for m in range(-1*m_range, m_range + 1):
            
            for n in range(l + 1, n_max + 1):
                Psi[(l, m)] -= np.sum(Bound_States[(n,l)].conj()*Psi[(l,m)])*Bound_States[(n,l)]
            
            COEF_Minor = {}
            for i in range(smallest_eigen_number):
                
                coef = np.exp(-1.j*Phase[l,i])* 1.j**l *np.sum(CWF_Psi[l,i].conj()*Psi[(l,m)])    

                COEF_Minor[i] = (coef.real, coef.imag)#coef
    
            COEF[str((l,m))] = COEF_Minor
    
    return COEF

def Coef_Organizer(COEF, CWF_Energy, smallest_eigen_number):
    COEF_Organized = {}
    k_array = {}

    for i in range(smallest_eigen_number):
        k_array[i] = np.sqrt(2.0*CWF_Energy[0,i])

    for i in range(smallest_eigen_number):
        COEF_Minor = {}
        for l in range(0, input_par["l_max"]+ 1):    
            m_range = min(l, input_par["m_max"])
            for m in range(-1*m_range, m_range + 1):
                COEF_Minor[str((l,m))] = COEF[str((l,m))][i]

        COEF_Organized[k_array[i]] = COEF_Minor



    return COEF_Organized, k_array

def Coef_Organizer_2(COEF, k_new):
    COEF_Organized = {}
    for i, k in enumerate(k_new):
        COEF_Minor = {}
        for l in range(0, input_par["l_max"]+ 1):    
            m_range = min(l, input_par["m_max"])
            for m in range(-1*m_range, m_range + 1):
                COEF_Minor[str((l,m))] = COEF[str((l,m))][i]

        COEF_Organized[round(k,5)] = COEF_Minor



    return COEF_Organized

def Coef_Interpolate(COEF, CWF_Energy, smallest_eigen_number):
    COEF_Major = {}
    k_array = np.zeros(smallest_eigen_number)

    k_min = 100
    for l in range(0, input_par["l_max"]+ 1):  
        for i in range(smallest_eigen_number):
            k_array[i] = np.sqrt(2.0*CWF_Energy[l,i])
        if k_min > k_array.max():
            k_min = k_array.max()

    k_new = np.linspace(0, k_min, smallest_eigen_number)

    print(k_new[1] - k_new[0])
    for l in range(0, input_par["l_max"]+ 1):    
        m_range = min(l, input_par["m_max"])
        for i in range(smallest_eigen_number):
            k_array[i] = np.sqrt(2.0*CWF_Energy[l,i])


        for m in range(-1*m_range, m_range + 1):
            COEF_Minor = list(COEF[str((l,m))].values())
          
            COEF_Minor = interpolate.splrep(k_array, COEF_Minor[:smallest_eigen_number], s=0)
            COEF_Minor = interpolate.splev(k_new, COEF_Minor, der=0)

            COEF_Major[str((l,m))] = COEF_Minor

            
    return COEF_Major, k_new

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

def PAD_Momentum(COEF, input_par):
    # print(COEF.keys())
    resolution = 0.025
    x_momentum = np.arange(-1. ,1. + resolution, resolution)
    z_momentum = np.arange(-1. ,1. + resolution, resolution)
    resolution = 0.05
    y_momentum = np.arange(-1. ,1. + resolution, resolution)
    

    pad_value = np.zeros((z_momentum.size,x_momentum.size))

    for i, px in enumerate(x_momentum):
        print(px)
        for j, pz in enumerate(z_momentum):
            pad_value_temp = 0.0
            for l, py in enumerate(y_momentum):
            
                k = np.sqrt(px*px + py*py + pz*pz)
                if k == 0:
                    continue
                
                if px > 0 and py > 0:
                    phi = np.arctan(py/px)
                elif px > 0 and py < 0:
                    phi = np.arctan(py/px) + 2*pi
                elif px < 0 and py > 0:
                    phi = np.arctan(py/px) + pi
                elif px < 0 and py < 0:
                    phi = np.arctan(py/px) + pi
                elif px == 0 and py == 0:
                    phi = 0
                elif px == 0 and py > 0:
                    phi = pi / 2
                elif px == 0 and py < 0:
                    phi = 3*pi / 2
                elif py == 0 and px > 0:
                    phi = 0
                elif py == 0 and px < 0:
                    phi = pi

                theta = np.arccos(pz/k)
                coef_dic = COEF[closest(list(COEF.keys()), k)]
                pad_value_temp +=  np.abs(K_Sphere(coef_dic, input_par, phi, theta))**2

            pad_value[j, i] = pad_value_temp[0][0]

    return pad_value, x_momentum, y_momentum

def Coef_Plotter(coef_main, COEF, CWF_Energy, smallest_eigen_number):
    k_max = 2
    dk = 0.01
    k_array = np.arange(dk, k_max + dk, dk)
    coef = np.zeros(len(k_array), dtype=complex)
    l = 5
    m = 0
    for i, k in enumerate(k_array): 
        k = round(k , 5)
        coef[i] = np.absolute(coef_main[str(k)][str((l,m))][0] + 1.0j*coef_main[str(k)][str((l,m))][1])

    plt.plot(k_array, coef.real, '.')

    COEF = list(COEF[str((l,m))].values())

    coef_2 = np.zeros(smallest_eigen_number , dtype=complex)
    k_array_2 = np.zeros(smallest_eigen_number)
    for i in range(smallest_eigen_number):
            k_array_2[i] = np.sqrt(2.0*CWF_Energy[l,i])
            coef_2[i] = np.absolute(COEF[i]) #np.absolute(COEF[i][0] + 1.0j*COEF[i][0])

    coef_2 = interpolate.splrep(k_array_2, coef_2, s=0)

    coef_2 = interpolate.splev(k_array, coef_2)
    
    plt.plot(k_array, coef_2.real, '+')

    plt.xlim(-0.1,1)
    plt.savefig("COEF_Old.png")
    plt.clf()


    # COEF_Organized, k_array = Coef_Organizer(COEF, CWF_Energy, smallest_eigen_number)
    
    
    # # with open("PAD.json") as file:
    # #     coef_main = json.load(file)

    # # Coef_Plotter(coef_main, COEF, CWF_Energy, smallest_eigen_number)
    
    # # COEF_Major, k_new = Coef_Interpolate(COEF, CWF_Energy, smallest_eigen_number)
    
    # # COEF_Organized = Coef_Organizer_2(COEF, k_new)


    # with open("PAD_New.json", 'w') as file:
    #     json.dump(COEF_Organized, file)
  
    # with open("PAD_New.json") as file:
    #     COEF_Organized = json.load(file)