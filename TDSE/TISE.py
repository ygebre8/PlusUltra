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
    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    EV_Solver.setOperators(Hamiltonian)
    # EV_Solver.setType(SLEPc.EPS.Type.JD)
    EV_Solver.setProblemType(SLEPc.EPS.ProblemType.NHEP) 
    EV_Solver.setTolerances(input_par["tolerance"], PETSc.DECIDE) 
    EV_Solver.setWhichEigenpairs(EV_Solver.Which.SMALLEST_REAL) 
    size_of_matrix = PETSc.Mat.getSize(Hamiltonian) 
    dimension_size = int(size_of_matrix[0]) * 0.1 
    EV_Solver.setDimensions(number_of_eigenvalues, PETSc.DECIDE, dimension_size) 
    EV_Solver.solve() 
   
    num_of_cont = 0
    if rank == 0:
        print("Number of eigenvalues requested and converged")
        print(number_of_eigenvalues, EV_Solver.getConverged(), "\n")

    converged = EV_Solver.getConverged()
    for i in range(converged):
        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)

        if eigen_state.real < 0:
            eigen_vector.setName("BS_Psi_" + str(l) + "_" + str(i + l + 1)) 
            Viewer.view(eigen_vector)
            
            energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
            energy.setValue(0,eigen_state)
            energy.setName("BS_Energy_" + str(l) + "_" + str(i + l + 1))
            energy.assemblyBegin()
            energy.assemblyEnd()
            Viewer.view(energy)

        else:
            eigen_vector.setName("CS_Psi_" + str(l) + "_" + str(num_of_cont)) 
            Viewer.view(eigen_vector)
            
            energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
            energy.setValue(0,eigen_state)
            energy.setName("CS_Energy_" + str(l) + "_" + str(num_of_cont))

            energy.assemblyBegin()
            energy.assemblyEnd()
            Viewer.view(energy)

            num_of_cont += 1

  
    return num_of_cont
    
def Build_Hamiltonian_Fourth_Order(potential, grid, input_par):
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

def Build_Hamiltonian_Second_Order(potential, grid, input_par):
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

def TISE(input_par):

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    
    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(input_par["Target_File"], mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    smal_cont_state = 10e5

    if input_par["calc_cont_states"] == 0: 
        l_range = input_par["n_max"]
    if input_par["calc_cont_states"] == 1:
        l_range = input_par["l_max"] + 1

    for l in range(0, l_range):
        if rank == 0:
            print("Calculating the eigenstates for l = " + str(l) + "\n")
    
        potential = eval("Pot." + input_par["potential"] + "(grid, l)")
        Hamiltonian = eval("Build_Hamiltonian_" + input_par["order"] + "_Order(potential, grid, input_par)")       

        if input_par["calc_cont_states"] == 0: 
            num_of_cont_calc = Eigen_Value_Solver(Hamiltonian, input_par["n_max"] - l, input_par, l, ViewHDF5)
        if input_par["calc_cont_states"] == 1:
            num_of_cont_calc = Eigen_Value_Solver(Hamiltonian, input_par["num_of_cont_states"], input_par, l, ViewHDF5)
        if num_of_cont_calc < smal_cont_state:
            smal_cont_state = num_of_cont_calc

        if rank == 0:
            print("Finished calculation for l = " + str(l) + "\n" , "\n")

    CSC = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
    CSC.setValue(0,smal_cont_state)
    CSC.setName("CSC")
    CSC.assemblyBegin()
    CSC.assemblyEnd()
    ViewHDF5.view(CSC)
    ViewHDF5.destroy()
        
if __name__=="__main__":
    start_time = time.time()
    input_par = Mod.Input_File_Reader("input.json")
    TISE(input_par)
    total_time = round((time.time() - start_time) / 60, 3)
    
    if rank == 0:    
        print("Total time taken for calculating eigenstates is ", total_time)