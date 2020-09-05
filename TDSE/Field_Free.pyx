if True:
    import numpy as np
    import sys
    from numpy import pi
    from math import floor
    import Module as Mod 
    import Potential as Pot
         
if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    petsc4py.init(comm=PETSc.COMM_WORLD)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

def Build_FF_Hamiltonian_Fourth_Order(input_par):

    cdef float h2
    cdef int grid_size, ECS, l_block, grid_idx, ECS_idx

    index_map_l_m, index_map_box = Mod.Index_Map(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size
    matrix_size = grid_size * len(index_map_box)
    h2 = input_par["grid_spacing"] * input_par["grid_spacing"]

    if input_par["ECS_region"] < 1.00 and input_par["ECS_region"] > 0.00:
        ECS_idx = np.where(grid > grid[-1] * input_par["ECS_region"])[0][0]
    elif input_par["ECS_region"] == 1.00:
        ECS_idx = grid_size
        if rank == 0:
            print("No ECS applied for this run \n")
    else:
        if rank == 0:
            print("ECS region has to be between 0.0 and 1.00\n")
            exit() 
            
    def Fourth_Order_Stencil():
        x_2 = 0.25*(2j - 3*np.exp(3j*pi/4)) / (1 + 2j + 3*np.exp(1j*pi/4))
        x_1 = (-2j + 6*np.exp(3j*pi/4)) / (2 + 1j + 3*np.exp(1j*pi/4))
        x = 0.25*(-2 + 2j - 9*np.exp(3j*pi/4))
        x__1 = (2 + 2j - 6*np.sqrt(2)) / (3 + 1j + 3*np.sqrt(2))
        x__2 = 0.25*(-2 -2j + 3*np.sqrt(2)) / (3 - 1j + 3*np.sqrt(2))
        return (x__2, x__1, x, x_1, x_2)
    
    ECS_Stencil = Fourth_Order_Stencil()

    potential = np.zeros(shape=(input_par["l_max"] + 1, grid_size))
    for i, l in enumerate(potential):
        potential[i] = eval("Pot." + input_par["potential"] + "(grid, i)")


    FF_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=5, comm=PETSc.COMM_WORLD)
    istart, iend = FF_Hamiltonian.getOwnershipRange()
    for i in range(istart, iend):
        grid_idx = i % grid_size 
        l_block = index_map_l_m[floor(i/grid_size)][0]
        
      
        if grid_idx < ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + (15.0/ 12.0)/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
            if grid_idx < grid_size - 2:
                FF_Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)
        
        if grid_idx == ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + ECS_Stencil[2]/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, ECS_Stencil[1]/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, ECS_Stencil[0]/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, ECS_Stencil[3]/h2)
            if grid_idx < grid_size - 2:
                FF_Hamiltonian.setValue(i, i+2, ECS_Stencil[4]/h2)

        if grid_idx > ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + (15.0/ 12.0)* -1.0j/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, (-2.0/3.0) * -1.0j/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, (1.0/24.0) * -1.0j/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, (-2.0/3.0) * -1.0j/h2)
            if grid_idx < grid_size - 2:
                FF_Hamiltonian.setValue(i, i+2, (1.0/24.0) * -1.0j/h2)

    for i in np.arange(0, matrix_size, grid_size):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        
        FF_Hamiltonian.setValue(i, i, potential[l_block][0] + (20.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+1, (-6.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+2, (-4.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+3, (1.0/24.0)/h2)

        j = i + (grid_size - 1)
        FF_Hamiltonian.setValue(j,j, potential[l_block][grid_size - 1] + (20.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 1, (-6.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 2, (-4.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 3, (1.0/24.0) * -1.0j/h2)

    FF_Hamiltonian.assemblyBegin()
    FF_Hamiltonian.assemblyEnd()
    return FF_Hamiltonian  

def Build_FF_Hamiltonian_Second_Order(input_par):

    cdef double h2
    cdef int grid_size, l_block, grid_idx, ECS_idx

    index_map_l_m, index_map_box = Mod.Index_Map(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size
    matrix_size = grid_size * len(index_map_box)
    h2 = input_par["grid_spacing"] * input_par["grid_spacing"]
    
    if input_par["ECS_region"] < 1.00 and input_par["ECS_region"] > 0.00:
        ECS_idx = np.where(grid > grid[-1] * input_par["ECS_region"])[0][0]
    elif input_par["ECS_region"] == 1.00:
        ECS_idx = grid_size
        if rank == 0:
            print("No ECS applied for this run \n")
    else:
        if rank == 0:
                print("ECS region has to be between 0.0 and 1.00\n")
                exit() 

    potential = np.zeros(shape=(input_par["l_max"] + 1, grid_size))
    for i, l in enumerate(potential):
        potential[i] = eval("Pot." + input_par["potential"] + "(grid, i)")
        

    FF_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = FF_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        grid_idx = i % grid_size 
        l_block = index_map_l_m[floor(i/grid_size)][0]

        if grid_idx < ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + 1.0/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, -0.5/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, -0.5/h2)
        
        elif grid_idx > ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] - 1.0j/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, 0.5j/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, 0.5j/h2)

        else:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] +  np.exp(-1.0j*pi/4.0)/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, -1/(1 + np.exp(1.0j*pi/4.0))/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, -1*np.exp(-1.0j*pi/4.0)/ (1+np.exp(1.0j*pi/4.0))/h2)

 
    FF_Hamiltonian.assemblyBegin()
    FF_Hamiltonian.assemblyEnd()
    return FF_Hamiltonian  

def Build_Full_Hamiltonian(FF_Hamiltonian, Int_Hamiltonian_x, Int_Hamiltonian_y, Int_Hamiltonian_z, Right_Circular_Matrix, Left_Circular_Matrix, input_par):
    matrix_size = FF_Hamiltonian.getSize()[0]
    if input_par["gauge"] == "Length":
        Full_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=6, comm=PETSc.COMM_WORLD)
    if input_par["gauge"] == "Velocity":
        Full_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=12, comm=PETSc.COMM_WORLD)

    FF_Hamiltonian.copy(Full_Hamiltonian, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    
    if Int_Hamiltonian_x != None:
        Full_Hamiltonian.axpy(0.0, Int_Hamiltonian_x, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if Int_Hamiltonian_y != None:
        Full_Hamiltonian.axpy(0.0, Int_Hamiltonian_y, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if Int_Hamiltonian_z != None:
        Full_Hamiltonian.axpy(0.0, Int_Hamiltonian_z, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if Right_Circular_Matrix != None:
        Full_Hamiltonian.axpy(0.0, Right_Circular_Matrix, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if Left_Circular_Matrix != None:
        Full_Hamiltonian.axpy(0.0, Left_Circular_Matrix, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

    
    Full_Hamiltonian.assemblyBegin()
    Full_Hamiltonian.assemblyEnd()
    return Full_Hamiltonian


   