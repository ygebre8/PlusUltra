if True:
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import sys
    import json
    import h5py
    import os
    import Potential as Pot
    import Module as Mod
    import TISE as TS
    from scipy.special import sph_harm
    from scipy.interpolate import interp1d

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

def Eigen_Value_Solver(Hamiltonian, l, k, Viewer, tol):
    number_of_eigenvalues = 1
    energy_target = k*k / 2
    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    EV_Solver.setOperators(Hamiltonian) ##pass the hamiltonian to the 
    # EV_Solver.setType(SLEPc.EPS.Type.JD)
    EV_Solver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    EV_Solver.setTolerances(tol, PETSc.DECIDE)
    EV_Solver.setWhichEigenpairs(EV_Solver.Which.TARGET_REAL)
    EV_Solver.setTarget(energy_target)
    size_of_matrix = PETSc.Mat.getSize(Hamiltonian)
    dimension_size = int(size_of_matrix[0]) * 0.1
    EV_Solver.setDimensions(number_of_eigenvalues, PETSc.DECIDE, dimension_size) 
    EV_Solver.solve() ##solve the eigenvalue problem
    eigen_states = [[],[]]
   
    for i in range(number_of_eigenvalues):
        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)
        eigen_vector.setName("Psi_" + str(k) + "_" + str(l)) 

        Viewer.view(eigen_vector)
        
        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        k_cal = np.sqrt(2*eigen_state)
        energy.setValue(0,k_cal)

        energy.setName("Energy_" + str(k) + "_" + str(l))
        energy.assemblyBegin()
        energy.assemblyEnd()

        Viewer.view(energy)  

def Continuum_Wavefuncion_Maker(input_par, k_max, dk):

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5("Continum.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)


    for k in np.arange(dk, k_max + dk, dk): 
        k = round(k ,5)
        if rank == 0:
            print(k)

        for l in range(0, input_par["l_max"]+ 1):

            if rank == 0:
                print("Calculating the eigenstates for l = " + str(l) + "\n")
        
            potential = eval("Pot." + input_par["potential"] + "(grid, l)")
            Hamiltonian = eval("TS.Build_Hamiltonian_" + input_par["order"] + "_Order(potential, grid, input_par)")        
            Eigen_Value_Solver(Hamiltonian, l, k, ViewHDF5, 1e-12)

            if rank == 0:
                print("Finished calculation for l = " + str(l) + "\n" , "\n")

    ViewHDF5.destroy()

def Continuum_Wavefuncion_Reader(input_par, k_max, dk, c_file_name, z = 1):
    CWF = {}
    phase = {}
    k_value = {}

    Target_file =h5py.File(c_file_name)
    
    dr = input_par["grid_spacing"]
    r = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    r_val = r[-2]

    for k in np.arange(dk, k_max + dk, dk): 
        k = round(k ,5)
        if rank == 0:
            print(k)

        for l in range(0, input_par["l_max"]+ 1):

            dataset_name = "Energy_" + str(k) + "_" + str(l)
            k_value[k,l] = Target_file[dataset_name]
            k_value[k,l] = np.array(k_value[k, l][:,0] + 1.0j*k_value[k, l][:,1])

            dataset_name = "Psi_" + str(k) + "_" + str(l)
            CWF[k, l] = Target_file[dataset_name]
            CWF[k, l] = np.array(CWF[k, l][:,0] + 1.0j*CWF[k, l][:,1])

            norm = np.array(CWF[k,l]).max()
            CWF[k, l] /= norm
        
        
            coul_wave_r = -1*np.linalg.norm(CWF[k, l][-2])
            dcoul_wave_r = -1*np.linalg.norm((CWF[k, l][-1]-CWF[k, l][-3])/(2*dr))

            # norm = np.sqrt(np.abs(coul_wave_r)**2+(np.abs(dcoul_wave_r)/(k+z/(k*r_val)))**2)
        

        
            phase[k, l] = np.angle((1.j*coul_wave_r + dcoul_wave_r/(k+z/(k*r_val))) /
                        (2*k*r_val)**(1.j*z/k)) - k*r_val + l*np.pi/2

    

    return phase, CWF, k_value

def K_Sphere(input_par, k, phase, CWF, Psi, Bound_States):
    coef_dic = {}
    for l in range(0, input_par["l_max"]+ 1):    
        n_max = input_par["n_max"]
        m_range = min(l, input_par["m_max"])
        for m in range(-1*m_range, m_range + 1):
            for n in range(l + 1, n_max + 1):
                Psi[(l, m)] -= np.sum(Bound_States[(n,l)].conj()*Psi[(l,m)])*Bound_States[(n,l)]
        
            coef = np.exp(-1.j*phase[k, l])*1.j**l * np.sum(CWF[k, l].conj()*Psi[(l,m)])
            coef_dic[str((l,m))] = (coef.real, coef.imag)

    return coef_dic

def Coef_Fun(k_value, coef_main, k_max, dk):
    coef_corr_main = {}
    k_array = np.arange(dk, k_max + dk, dk)
    coef = np.zeros(len(k_array))
    k_corr = np.zeros(len(k_array))
    for l in range(0, input_par["l_max"]+ 1):    
        m_range = min(l, input_par["m_max"])
        for m in range(-1*m_range, m_range + 1):
            for i, k in enumerate(k_array): 
                k = round(k ,5)
                coef[i] = np.abs(coef_main[str(k)][str((l,m))][0] + 1.0j*coef_main[str(k)][str((l,m))][1])
                k_corr[i] = np.abs(k_value[k,l])
            f2 = interp1d(k_corr, coef, kind='cubic', fill_value='extrapolate')
            coef_corr_main[l,m] = f2(k_array)

    return coef_corr_main

def Coef_Reorder(coef_corr_main, k_max, dk):
    coef_main = {}
    k_array = np.arange(dk, k_max + dk, dk)
    for i, k in enumerate(k_array): 
        k = round(k ,5)
        coef_dic = {}
        for l in range(0, input_par["l_max"]+ 1):    
            m_range = min(l, input_par["m_max"])
            for m in range(-1*m_range, m_range + 1):
                coef_dic[str((l,m))] = coef_corr_main[l,m][i]
        coef_main[str(k)] = coef_dic

    return coef_main

if __name__=="__main__":

    input_par = Mod.Input_File_Reader("input.json")
    # Psi = Psi_Reader(input_par)
    # Bound_States = Bound_Reader(input_par)

    k_max = .9
    dk = 0.01
    
    Continuum_Wavefuncion_Maker(input_par,  k_max, dk)
    # c_file_name = "Continum_More.h5"
    # phase, CWF, k_value = Continuum_Wavefuncion_Reader(input_par, k_max, dk, c_file_name, z = 1)
    
    # coef_main = {}
    # for k in np.arange(dk, k_max + dk, dk): 
    #     k = round(k , 5)
    #     print(k)
    #     coef_dic = K_Sphere(input_par, k, phase, CWF, Psi, Bound_States)
    #     coef_main[str(k)] = coef_dic

    # coef_corr_main = Coef_Fun(k_value, coef_main, k_max, dk)

    # coef_main = Coef_Reorder(coef_corr_main, k_max, dk)

    # with open("PAD_New.json", 'w') as file:
    #     json.dump(coef_main, file)
        