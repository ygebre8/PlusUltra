if True:
    import sys
    import time 
    import Module as Mod 
    import Potential as Pot

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

def Eigen_Value_Solver(Hamiltonian, number_of_eigenvalues, input_par, l, Viewer):
    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD) #create the eigen value solver object
    EV_Solver.setOperators(Hamiltonian) ##pass the hamiltonian to the  ex solver
    # EV_Solver.setType(SLEPc.EPS.Type.JD) ##what kind of solver to use for the problem
    EV_Solver.setProblemType(SLEPc.EPS.ProblemType.NHEP) ## problem type, set to non hermitian sice fourth order need a stencil correction
    EV_Solver.setTolerances(input_par["tolerance"], PETSc.DECIDE) ## tolerance for the solver, the error tolerance
    EV_Solver.setWhichEigenpairs(EV_Solver.Which.SMALLEST_REAL) #looking for the smallest real eigen values
    size_of_matrix = PETSc.Mat.getSize(Hamiltonian) ## getting matrix size for setting the dimension of the solver
    dimension_size = int(size_of_matrix[0]) * 0.1 ## here the 0.1 is arbritary and just seems to be a good value
    EV_Solver.setDimensions(number_of_eigenvalues, PETSc.DECIDE, dimension_size) 
    EV_Solver.solve() ##solve the eigenvalue problem
   

    if rank == 0:
        print("Number of eigenvalues requested and converged")
        print(number_of_eigenvalues, EV_Solver.getConverged(), "\n")
   
    num_of_cont_states_for_l = 0
    for i in range(number_of_eigenvalues):
        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)

        if eigen_state.real < 0:
            ## storing the eigen vectors (wavefunctions) as BS_Psi_l_n where n and l are the quantum numbers
            eigen_vector.setName("BS_Psi_" + str(l) + "_" + str(i + l + 1)) 
            Viewer.view(eigen_vector)
            
            ## storing the eigen values (energies) as BS_Energy_l_n where n and l are the quantum numbers
            energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
            energy.setValue(0,eigen_state)
            energy.setName("BS_Energy_" + str(l) + "_" + str(i + l + 1))
            energy.assemblyBegin()
            energy.assemblyEnd()
            Viewer.view(energy)

        else:
            ## storing the eigen vectors (wavefunctions) as CS_Psi_l_i where l is the quantum number and i is just a simple index for counting 
            eigen_vector.setName("CS_Psi_" + str(l) + "_" + str(num_of_cont_states_for_l)) 
            Viewer.view(eigen_vector)
            
            energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
            energy.setValue(0,eigen_state)
            
            ## storing the eigen values (energies) as Energy_l_i where l is the quantum number and i is just a simple index for counting 
            energy.setName("CS_Energy_" + str(l) + "_" + str(num_of_cont_states_for_l))

            energy.assemblyBegin()
            energy.assemblyEnd()
            Viewer.view(energy)

            num_of_cont_states_for_l += 1

    if input_par["calculate_cont_states"] == 1:
        return num_of_cont_states_for_l
    
def Build_Hamiltonian_Fourth_Order_BS(potential, grid, input_par):
    matrix_size = grid.size
    h2 = input_par["grid_spacing"]*input_par["grid_spacing"] ## dr^2, where dr is the grid spacing

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=5, comm=PETSc.COMM_WORLD) ## create the hamiltonian object
    istart, iend = Hamiltonian.getOwnershipRange() ## for each core get the row that it owns (the matrix is split be)
    for i  in range(istart, iend):
        Hamiltonian.setValue(i, i, potential[i] + (15.0/ 12.0)/h2)
        
        if i >=  1:
            Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
        if i >= 2:
            Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
        if i < grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
        if i < grid.size - 2:
            Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)

   
    Hamiltonian.setValue(0,0, potential[0]  + (20.0/24.0)/h2)
    Hamiltonian.setValue(0,1, (-6.0/24.0)/h2)
    Hamiltonian.setValue(0,2, (-4.0/24.0)/h2)
    Hamiltonian.setValue(0,3, (1.0/24.0)/h2)
    
    j = grid.size - 1
    Hamiltonian.setValue(j,j, potential[j] + (20.0/24.0)/h2)
    Hamiltonian.setValue(j,j - 1, (-6.0/24.0)/h2)
    Hamiltonian.setValue(j,j - 2, (-4.0/24.0)/h2)
    Hamiltonian.setValue(j,j - 3, (1.0/24.0)/h2)

    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def Build_Hamiltonian_Second_Order_BS(potential, grid, input_par):
    matrix_size = grid.size
    h2 = input_par["grid_spacing"]*input_par["grid_spacing"]

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()
    for i  in range(istart, iend):

        Hamiltonian.setValue(i, i, potential[i] +  1.0/h2)    
        if i >=  1:
            Hamiltonian.setValue(i, i-1, (-1.0/2.0)/h2)
        if i < grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-1.0/2.0)/h2)
    
    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def Build_Hamiltonian_Second_Order_CS(potential, grid, input_par):
    matrix_size = grid.size
    h2 = input_par["grid_spacing"]*input_par["grid_spacing"]

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()
    for i  in range(istart, iend):

        Hamiltonian.setValue(i, i, potential[i] +  1.0/h2)    
        if i >=  1:
            Hamiltonian.setValue(i, i-1, (-1.0/2.0)/h2)
        if i < grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-1.0/2.0)/h2)
    
    j = grid.size - 1
    Hamiltonian.setValue(j,j, potential[j] + (-1.0/2.0)/h2)
    Hamiltonian.setValue(j,j - 1, 1.0/h2)
    Hamiltonian.setValue(j,j - 2, (-1.0/2.0)/h2)


    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def Build_Hamiltonian_Fourth_Order_CS(potential, grid, input_par):
    return 0

def Calculate_Bound_States(input_par):
    
    if rank == 0:
        start_time = time.time()
        print("\n")

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    
    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(input_par["Target_File"], mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    for l in range(0, input_par["n_max"]):
        if rank == 0:
            print("Calculating the eigenstates for l = " + str(l) + "\n")
    
        potential = eval("Pot." + input_par["potential"] + "(grid, l)")
        Hamiltonian = eval("Build_Hamiltonian_" + input_par["order"] + "_Order_BS(potential, grid, input_par)")        
        Eigen_Value_Solver(Hamiltonian, input_par["n_max"] - l, input_par, l, ViewHDF5)

        if rank == 0:
            print("Finished calculation for l = " + str(l) + "\n" , "\n")

    if rank == 0:
        total_time = (time.time() - start_time) / 60
        print("Total time taken for calculating eigenstates is " + str(round(total_time, 3)))

    ViewHDF5.destroy()

def Calculate_Continuum_States(input_par):

    if rank == 0:
        start_time = time.time()
        print("\n")

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(input_par["Target_File"], mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    smallest_cont_state = 10e5
    for l in range(0, input_par["l_max"]+ 1):

        if rank == 0:
            print("Calculating the eigenstates for l = " + str(l) + "\n")
    
        potential = eval("Pot." + input_par["potential"] + "(grid, l)")
        Hamiltonian = eval("Build_Hamiltonian_" + input_par["order"] + "_Order_CS(potential, grid, input_par)")        
        num_of_cont_states_calculated = Eigen_Value_Solver(Hamiltonian, input_par["num_of_cont_states"], input_par, l, ViewHDF5)

        if num_of_cont_states_calculated < smallest_cont_state:
            smallest_cont_state = num_of_cont_states_calculated

        if rank == 0:
            print("Finished calculation for l = " + str(l) + "\n" , "\n")

    if rank == 0:
        total_time = (time.time() - start_time) / 60
        print("Total time taken for calculating eigenstates is " + str(round(total_time, 3)))

    CSC = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
    CSC.setValue(0,smallest_cont_state)
    CSC.setName("CSC")
    CSC.assemblyBegin()
    CSC.assemblyEnd()
    ViewHDF5.view(CSC)

    ViewHDF5.destroy()

def TISE(input_par):
    if input_par["calculate_cont_states"] == 1:
        Calculate_Continuum_States(input_par)
    elif input_par["calculate_cont_states"] == 0:
        Calculate_Bound_States(input_par)
    else:
        if rank == 0:
            print("Error: 'calculate_cont_states' has to be 1 or 0 for True or False respectively.")
        exit()
        
if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    TISE(input_par)

    

