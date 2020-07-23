if True:
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import json
    import h5py
    import sys
    from scipy.special import sph_harm
    from scipy import interpolate
    from numpy import pi
    
    import PAD_Module as PAD_Mod

    sys.path.append('/home/becker/yoge8051/Research/PlusUltra/TDSE')
    import TISE as TS
    import Module as Mod
    import Potential as Pot

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

def Eigen_Value_Solver(Hamiltonian, number_of_eigenvalues, input_par, l, Viewer):
    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    EV_Solver.setOperators(Hamiltonian) ##pass the hamiltonian to the 
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

def Phase_Calculator(input_par, CWF_Energy, CWF_Psi):

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

            coul_wave_r = CWF_Psi[l,i][-2]
            dcoul_wave_r = (CWF_Psi[l,i][-1]-CWF_Psi[l,i][-3])/(2*dr)
            k = np.sqrt(2.0*CWF_Energy[l,i])

            Phase[l,i] = np.angle((1.j*coul_wave_r + dcoul_wave_r/(k+z/(k*r_val))) / (2*k*r_val)**(1.j*z/k)) - k*r_val + l*np.pi/2

    
    return Phase, smallest_eigen_number

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

                COEF_Minor[i] = coef
    
            COEF[str((l,m))] = COEF_Minor
    
    return COEF

def Coef_Interpolate(COEF, CWF_Energy, smallest_eigen_number):

    COEF_Major = {}
    k_array = np.zeros(smallest_eigen_number)

    for l in range(0, input_par["l_max"]+ 1):  
        for i in range(smallest_eigen_number):
            k_array[i] = np.sqrt(2.0*CWF_Energy[l,i])

    print(k_array[-1])
    k_max = 2
    dk = 0.01

    k_new = np.arange(dk, k_max + dk, dk)

    for l in range(0, input_par["l_max"]+ 1):    
        m_range = min(l, input_par["m_max"])
        for i in range(smallest_eigen_number):
            k_array[i] = np.sqrt(2.0*CWF_Energy[l,i])


        for m in range(-1*m_range, m_range + 1):
            COEF_Minor_Real = np.array(list(COEF[str((l,m))].values())).real
            
            COEF_Minor_Real = interpolate.splrep(k_array, COEF_Minor_Real[:smallest_eigen_number], s=0)
            COEF_Minor_Real = interpolate.splev(k_new, COEF_Minor_Real, der=0)

            COEF_Minor_Imag = np.array(list(COEF[str((l,m))].values())).imag
            
            COEF_Minor_Imag = interpolate.splrep(k_array, COEF_Minor_Imag[:smallest_eigen_number], s=0)
            COEF_Minor_Imag = interpolate.splev(k_new, COEF_Minor_Imag, der=0)

            COEF_Major[str((l,m))] = COEF_Minor_Real + 1.0j*COEF_Minor_Imag

            
    return COEF_Major, k_new

def Coef_Organizer(COEF, CWF_Energy, k_array):
    COEF_Organized = {}

    for i, k in enumerate(k_array):
        COEF_Minor = {}
        for l in range(0, input_par["l_max"]+ 1):    
            m_range = min(l, input_par["m_max"])
            for m in range(-1*m_range, m_range + 1):
                COEF_Minor[str((l,m))] = COEF[str((l,m))][i]

        COEF_Organized[k] = COEF_Minor

    return COEF_Organized

def K_Sphere(coef_dic, input_par, phi, theta):
    theta, phi = np.meshgrid(theta, phi)
    out_going_wave = np.zeros(phi.shape, dtype=complex)
    for l in range(0, input_par["l_max"]+ 1):    
        m_range = min(l, input_par["m_max"])
        for m in range(-1*m_range, m_range + 1):
            coef = coef_dic[str((l,m))]#[0] + 1j*coef_dic[str((l,m))][1]
            out_going_wave += coef*sph_harm(m, l, phi, theta)

    return out_going_wave

def closest(lst, k): 
    return lst[min(range(len(lst)), key = lambda i: abs(float(lst[i])-k))] 

def PAD_Momentum(COEF, input_par):
    # print(COEF.keys())
    resolution = 0.01
    x_momentum = np.arange(-1.5 ,1.5 + resolution, resolution)
    y_momentum = np.arange(-1.5 ,1.5 + resolution, resolution)

    resolution = 0.05
    z_momentum = np.arange(-1.5 ,1.5 + resolution, resolution)
    
    

    pad_value = np.zeros((y_momentum.size,x_momentum.size))

    for i, px in enumerate(x_momentum):
        print(px)
        for j, py in enumerate(y_momentum):
            pad_value_temp = 0.0
            for l, pz in enumerate(z_momentum):
            
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

if __name__=="__main__":

    input_par = Mod.Input_File_Reader("input.json")
    number_of_eigenvalues = 2500

    
    # Continuum_Wavefuncion_Maker(input_par, number_of_eigenvalues)

    Psi = PAD_Mod.Psi_Reader(input_par)
    Bound_States = PAD_Mod.Bound_Reader(input_par)
    
    print("one")
    CWF_Energy, CWF_Psi = PAD_Mod.Continuum_Wavefuncion_Reader(input_par)

    Phase, smallest_eigen_number = Phase_Calculator(input_par, CWF_Energy, CWF_Psi)
    print("two")
    COEF = Coef_Calculator(input_par, Psi, Bound_States, CWF_Energy, CWF_Psi, Phase, smallest_eigen_number)

    COEF_Inter, k_array =  Coef_Interpolate(COEF, CWF_Energy, smallest_eigen_number)

    COEF_Organized = Coef_Organizer(COEF_Inter, CWF_Energy, k_array)
    print("three")

    pad_value, x_momentum, y_momentum = PAD_Momentum(COEF_Organized, input_par)
    pad_value = pad_value / pad_value.max()
    plt.imshow(pad_value, cmap='jet')#, interpolation="spline16")#, interpolation='nearest')
    plt.savefig("PAD_New.png")