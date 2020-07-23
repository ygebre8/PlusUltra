if True:
    import sys
    import time
    import json
    import H2_Module as Mod 
    import Potential as Pot
    from math import ceil, floor
    import numpy as np

if True:
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

def Eigen_Value_Solver(Hamiltonian, number_of_eigenvalues, input_par, m, Viewer):
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
   
    for i in range(number_of_eigenvalues):
        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)
        if rank == 0:
            print(eigen_state)
        eigen_vector.setName("Psi_" + str(m) + "_" + str(i)) 
        Viewer.view(eigen_vector)
        
        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        energy.setValue(0,eigen_state)
        
        energy.setName("Energy_" + str(m) + "_" + str(i))
        energy.assemblyBegin()
        energy.assemblyEnd()
        Viewer.view(energy)

def Build_Hamiltonian_Second_Order(input_par, m):

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * (input_par["l_max"] + 1)

    if input_par["l_max"] % 2 == 0:
        nnz = int(input_par["l_max"] / 2 + 1) + 2
    else:
        nnz = int(ceil(input_par["l_max"] / 2)) + 2

    h2 = input_par["grid_spacing"]*input_par["grid_spacing"]

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()

    with open("Nuclear_Electron_Int.json") as file:
        potential = json.load(file)

    for i  in range(istart, iend):
        l_block = floor(i/grid_size)
        grid_idx = i % grid_size
  
        Hamiltonian.setValue(i, i, 1.0/h2 + potential[str((m, l_block, l_block))][grid_idx] + 0.5*l_block*(l_block+1)*np.power(grid[grid_idx], -2.0))    
        if i >=  1:
            Hamiltonian.setValue(i, i-1, (-1.0/2.0)/h2)
        if i < grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-1.0/2.0)/h2)
    

        if l_block % 2 == 0:
            l_prime_list = range(0, input_par["l_max"] + 1, 2)
        else:
            l_prime_list = range(1, input_par["l_max"] + 1, 2)

        for l_prime in l_prime_list:
            if l_block == l_prime:
                continue

            col_idx = grid_size*l_prime + grid_idx
            Hamiltonian.setValue(i, col_idx, potential[str((m, l_block, l_prime))][grid_idx])

    for i in np.arange(0, matrix_size, grid_size):
        l_block = floor(i/grid_size)
        Hamiltonian.setValue(i, i, (1.0/2.0)/h2 + potential[str((m, l_block, l_block))][0] + 0.5*l_block*(l_block+1)*np.power(grid[0], -2.0))  


    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def Build_Hamiltonian_Fourth_Order(input_par, m):

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * (input_par["l_max"] + 1)

    if input_par["l_max"] % 2 == 0:
        nnz = int(input_par["l_max"] / 2 + 1) + 2
    else:
        nnz = int(ceil(input_par["l_max"] / 2)) + 2

    h2 = input_par["grid_spacing"]*input_par["grid_spacing"]

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()

    with open("Nuclear_Electron_Int.json") as file:
        potential = json.load(file)

    for i  in range(istart, iend):
        l_block = floor(i/grid_size)
        grid_idx = i % grid_size
        
        Hamiltonian.setValue(i, i, (15.0/ 12.0)/h2 + potential[str((m, l_block, l_block))][grid_idx]+ 0.5*l_block*(l_block+1)*np.power(grid[grid_idx], -2.0))    
        if i >=  1:
            Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
        if i >= 2:
            Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
        if i < grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
        if i < grid.size - 2:
            Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)


        if l_block % 2 == 0:
            l_prime_list = range(0, input_par["l_max"] + 1, 2)
        else:
            l_prime_list = range(1, input_par["l_max"] + 1, 2)

        for l_prime in l_prime_list:
            if l_block == l_prime:
                continue

            col_idx = grid_size*l_prime + grid_idx
            Hamiltonian.setValue(i, col_idx, potential[str((m, l_block, l_prime))][grid_idx])

    
    # for i in np.arange(0, matrix_size, grid_size):
    #     l_block = floor(i/grid_size)
        
    #     Hamiltonian.setValue(i, i, potential[str((m, l_block, l_block))][0] + (14.0/24.0)/h2 + 0.5*l_block*(l_block+1)*np.power(grid[0], -2.0))
    #     Hamiltonian.setValue(i, i+1, (-15.0/24.0)/h2)
    
    #     Hamiltonian.setValue(i+1, i, (-15.0/24.0)/h2)

     


    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    
  
    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(input_par["Target_File"], mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    if rank == 0:
        print("start")
    


    Hamiltonian = Build_Hamiltonian_Second_Order(input_par, 0)
    Eigen_Value_Solver(Hamiltonian, 3, input_par, 0, ViewHDF5)

    # Hamiltonian = Build_Hamiltonian_Second_Order(input_par, 1)
    # Eigen_Value_Solver(Hamiltonian, 4, input_par, 1, ViewHDF5)

    # Hamiltonian = Build_Hamiltonian_Fourth_Order(input_par, 0)
    # Eigen_Value_Solver(Hamiltonian, 3, input_par, 0, ViewHDF5)