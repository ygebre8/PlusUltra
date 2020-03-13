if True:
    import numpy as np
    import sys
    import json
    from math import floor
    sys.path.append('/home/becker/yoge8051/Research/PlusUltra/TDSE')
    import Module as Mod 
    import Coefficent_Calculator as CC
    import time as time_mod

         
if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    petsc4py.init(comm=PETSc.COMM_WORLD)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()




def Z_Matrix_L_Block(input_par):
  
    cdef double factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)

    Length_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=2, comm=PETSc.COMM_WORLD)
    istart, iend = Length_Gauge_Int_Hamiltonian.getOwnershipRange() 
    
    Coeff_Upper, Coeff_Lower = CC.Length_Gauge_Z_Coeff_Calculator(input_par)

    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        
        if l_block < input_par["l_max"]:
            col_idx = grid_size*index_map_box[(l_block + 1, m_block)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, col_idx, grid[grid_idx] * Coeff_Upper[l_block,m_block])
        
        
        if abs(m_block) < l_block and l_block > 0:
            col_idx = grid_size*index_map_box[(l_block - 1, m_block)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, col_idx, grid[grid_idx] * Coeff_Lower[l_block,m_block])
        
    Length_Gauge_Int_Hamiltonian.assemblyBegin()
    Length_Gauge_Int_Hamiltonian.assemblyEnd()
    return Length_Gauge_Int_Hamiltonian 

def X_Matrix_L_Block(input_par):

    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)


    Length_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Length_Gauge_Int_Hamiltonian.getOwnershipRange() 
    
    Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus = CC.Length_Gauge_X_Coeff_Calculator(input_par)

    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block < input_par["l_max"]:
           
            columon_idx = grid_size*index_map_box[(l_block + 1, m_block - 1)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Plus_Minus[l_block,m_block])
            
            columon_idx = grid_size*index_map_box[(l_block + 1, m_block + 1)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Plus_Plus[l_block,m_block])
        
        if l_block > 0:
            if -1*m_block < l_block - 1:
                columon_idx = grid_size*index_map_box[(l_block - 1, m_block - 1)] + grid_idx
                Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Minus_Minus[l_block,m_block])
            
            if m_block < l_block - 1:
                columon_idx = grid_size*index_map_box[(l_block - 1, m_block + 1)] + grid_idx
                Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Minus_Plus[l_block,m_block])

           
    Length_Gauge_Int_Hamiltonian.assemblyBegin()
    Length_Gauge_Int_Hamiltonian.assemblyEnd()
    return Length_Gauge_Int_Hamiltonian 

def Y_Matrix_L_Block(input_par):
    
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    Length_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Length_Gauge_Int_Hamiltonian.getOwnershipRange() 
   
    Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus = CC.Length_Gauge_Y_Coeff_Calculator(input_par)

    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size 


        if l_block < input_par["l_max"]:
            
            columon_idx = grid_size*index_map_box[(l_block + 1, m_block - 1)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Plus_Minus[l_block,m_block])
  
            columon_idx = grid_size*index_map_box[(l_block + 1, m_block + 1)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Plus_Plus[l_block,m_block])
        
        if l_block > 0:
            if -1*m_block < l_block - 1:

                columon_idx = grid_size*index_map_box[(l_block - 1, m_block - 1)] + grid_idx
                Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Minus_Minus[l_block,m_block])
            
            if m_block < l_block - 1:
            
                columon_idx = grid_size*index_map_box[(l_block - 1, m_block + 1)] + grid_idx
                Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Minus_Plus[l_block,m_block])

           
    Length_Gauge_Int_Hamiltonian.assemblyBegin()
    Length_Gauge_Int_Hamiltonian.assemblyEnd()
    return Length_Gauge_Int_Hamiltonian 

def Right_Circular_Matrix_L_Block(input_par):
    
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)

    Length_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=2, comm=PETSc.COMM_WORLD)
    istart, iend = Length_Gauge_Int_Hamiltonian.getOwnershipRange() 
    
    Coeff_Plus_Plus, Coeff_Minus_Plus = CC.Length_Gauge_Right_Coeff_Calculator(input_par)

    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block < input_par["l_max"]:
  
            columon_idx = grid_size*index_map_box[(l_block + 1, m_block + 1)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Plus_Plus[l_block,m_block])
        

        if l_block > 0:
            if m_block < l_block - 1:

                columon_idx = grid_size*index_map_box[(l_block - 1, m_block + 1)] + grid_idx
                Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Minus_Plus[l_block,m_block])

    Length_Gauge_Int_Hamiltonian.assemblyBegin()
    Length_Gauge_Int_Hamiltonian.assemblyEnd()
    return Length_Gauge_Int_Hamiltonian 

def Left_Circular_Matrix_L_Block(input_par):
    
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)

    Length_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Length_Gauge_Int_Hamiltonian.getOwnershipRange() 
    
    Coeff_Plus_Minus, Coeff_Minus_Minus = CC.Length_Gauge_Left_Coeff_Calculator(input_par)

    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if input_par["m_max"] == 0:
            Length_Gauge_Int_Hamiltonian.setValue(i, i, 0.0)
            continue

        if l_block < input_par["l_max"]:
            columon_idx = grid_size*index_map_box[(l_block + 1, m_block - 1)] + grid_idx
            Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Plus_Minus[l_block,m_block])
        
        if l_block > 0:
            if -1*m_block < l_block - 1: 
               
                columon_idx = grid_size*index_map_box[(l_block - 1, m_block - 1)] + grid_idx
                Length_Gauge_Int_Hamiltonian.setValue(i, columon_idx, grid[grid_idx] * Coeff_Minus_Minus[l_block,m_block])
                     
    Length_Gauge_Int_Hamiltonian.assemblyBegin()
    Length_Gauge_Int_Hamiltonian.assemblyEnd()
    return Length_Gauge_Int_Hamiltonian 
