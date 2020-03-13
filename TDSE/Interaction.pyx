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



def Velocity_Gauge_Rigth_Coeff_Calculator(input_par):
    Coeff_Plus_Plus = {}
    Coeff_Minus_Plus = {}
    cdef int l, m 
    for l in np.arange(input_par["l_max"] + 1):
        for m in np.arange(-1*l, l+1):
            Coeff_Plus_Plus[l,m] = 1.0j*np.sqrt((l+m+1)*(l+m+2) / (4*pow(l+1, 2.0) - 1))
            Coeff_Minus_Plus[l,m] = 1.0j*np.sqrt((l-m)*(l-m-1) / (4*pow(l, 2.0) - 1))        

    return Coeff_Plus_Plus, Coeff_Minus_Plus

def Velocity_Gauge_Left_Coeff_Calculator(input_par):
    Coeff_Plus_Minus = {}
    Coeff_Minus_Minus = {}
    cdef int l, m 
    for l in np.arange(input_par["l_max"] + 1):
        for m in np.arange(-1*l, l+1):
            Coeff_Plus_Minus[l,m] = 1.0j*np.sqrt((l-m+1)*(l-m+2)/(4*pow(l+1, 2.0) - 1))
            Coeff_Minus_Minus[l,m] = 1.0j*np.sqrt((l+m)*(l+m-1)/(4*pow(l, 2.0) - 1))

    return  Coeff_Plus_Minus, Coeff_Minus_Minus
def Length_Gauge_Z_Matrix_L_Block(input_par):
  
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

def Length_Gauge_X_Matrix_L_Block(input_par):

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

def Length_Gauge_Y_Matrix_L_Block(input_par):
    
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

def Length_Gauge_Right_Circular_Matrix_L_Block(input_par):
    
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

def Length_Gauge_Left_Circular_Matrix_L_Block(input_par):
    
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

def Velocity_Gauge_Z_Matrix_L_Block(input_par):

    cdef double h2
    cdef complex factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"] * 2.0

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=6, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block < input_par["l_max"]:

            col_idx = index_map_box[(l_block + 1, m_block)]*grid_size + grid_idx
            factor = 1.0j*np.sqrt((pow(l_block + 1 , 2.0) - pow(m_block, 2.0)) / (4*pow(l_block + 1, 2.0) - 1))
    
            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, factor *-1*(l_block+1)/grid[grid_idx])

            if grid_idx < grid_size - 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*factor/h2)

            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, factor/h2)

        if abs(m_block) < l_block and l_block > 0:

            col_idx = index_map_box[(l_block - 1, m_block)]*grid_size + grid_idx
            factor = 1.0j*np.sqrt((pow(l_block, 2.0) - pow(m_block, 2.0)) / (4*pow(l_block, 2.0) - 1))

            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, factor *l_block/grid[grid_idx])

            if grid_idx < grid_size - 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*factor/h2)

            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, factor/h2)

    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
    return Velocity_Gauge_Int_Hamiltonian

def Velocity_Gauge_X_Matrix_L_Block(input_par):
    
    cdef double h2
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"] * 2.0
    
    start_time = time_mod.time()
    Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus = CC.Velocity_Gauge_X_Coeff_Calculator(input_par)
    current_time = time_mod.time() - start_time

    current_time = time_mod.time()
    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=12, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 

    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size 

        if l_block < input_par["l_max"]:

            col_idx = index_map_box[(l_block+1, m_block-1)]*grid_size + grid_idx
            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Minus[l_block,m_block]*-1*(l_block+1)/grid[grid_idx])
            if grid_idx < grid_size - 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Plus_Minus[l_block,m_block]/h2)
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Plus_Minus[l_block,m_block]/h2)

            col_idx = index_map_box[(l_block+1, m_block+1)]*grid_size + grid_idx
            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Plus[l_block,m_block]*-1*(l_block+1)/grid[grid_idx])
            if grid_idx < grid_size - 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Plus_Plus[l_block,m_block]/h2)
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Plus_Plus[l_block,m_block]/h2)
        
        if l_block > 0:
            if -1*m_block < l_block-1:

                col_idx = index_map_box[(l_block-1, m_block-1)]*grid_size + grid_idx
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Minus[l_block,m_block]*l_block/grid[grid_idx])
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Minus_Minus[l_block,m_block]/h2)
                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Minus_Minus[l_block,m_block]/h2)
                
            if m_block < l_block-1:
                
                col_idx = index_map_box[(l_block-1, m_block+1)]*grid_size + grid_idx
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Plus[l_block,m_block]*l_block/grid[grid_idx])
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Minus_Plus[l_block,m_block]/h2)
                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Minus_Plus[l_block,m_block]/h2)



    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
    return Velocity_Gauge_Int_Hamiltonian 

   
def Velocity_Gauge_Y_Matrix_L_Block(input_par):
    
    cdef double h2
    cdef int l_block, m_block, grid_idx, col_idx, grid_size
    
    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"] *2.0

    Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus = CC.Velocity_Gauge_Y_Coeff_Calculator(input_par)

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=12, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size 


        if l_block < input_par["l_max"]:

            col_idx = index_map_box[(l_block+1, m_block-1)]*grid_size + grid_idx  
            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Minus[l_block,m_block]*-1*(l_block+1)/grid[grid_idx])
            if grid_idx < grid_size - 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Plus_Minus[l_block,m_block]/h2)
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Plus_Minus[l_block,m_block]/h2)


            col_idx = index_map_box[(l_block+1, m_block+1)]*grid_size + grid_idx
            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Plus[l_block,m_block]*-1*(l_block+1)/grid[grid_idx])
            if grid_idx < grid_size - 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Plus_Plus[l_block,m_block]/h2)
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Plus_Plus[l_block,m_block]/h2)
        
        if l_block > 0:
            if -1*m_block < l_block-1:

                col_idx = index_map_box[(l_block-1, m_block-1)]*grid_size + grid_idx
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Minus[l_block,m_block] * l_block/grid[grid_idx])
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Minus_Minus[l_block,m_block]/h2)
                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Minus_Minus[l_block,m_block]/h2)
                
            if m_block < l_block - 1:
                
                col_idx = index_map_box[(l_block-1, m_block+1)]*grid_size + grid_idx
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Plus[l_block,m_block]* l_block/grid[grid_idx])
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, -1*Coeff_Minus_Plus[l_block,m_block]/h2)
                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, Coeff_Minus_Plus[l_block,m_block]/h2)

           
    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
    return Velocity_Gauge_Int_Hamiltonian 

def Velocity_Gauge_Right_Circular_Matrix_L_Block(input_par):

    cdef double h2
    cdef complex factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"]*2.0

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=5, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 

    Coeff_Plus_Plus, Coeff_Minus_Plus = Velocity_Gauge_Rigth_Coeff_Calculator(input_par)
    
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block > 0:
            if m_block < l_block - 1:
                col_idx = index_map_box[(l_block - 1, m_block + 1)]*grid_size + grid_idx                

                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Minus_Plus[l_block, m_block]/h2)

                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Plus[l_block, m_block] * -1*l_block/grid[grid_idx])
                
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Minus_Plus[l_block, m_block]/h2)

        if l_block < input_par["l_max"]:
            col_idx = index_map_box[(l_block + 1, m_block + 1)]*grid_size + grid_idx           
         
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Plus_Plus[l_block, m_block]/h2)

            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Plus[l_block, m_block] *(l_block + 1)/grid[grid_idx])

            #if grid_idx < grid_size- 1:
             #   Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Plus_Plus[l_block, m_block]/h2)
            
                

    
    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
   
    
    return Velocity_Gauge_Int_Hamiltonian 

def Velocity_Gauge_Left_Circular_Matrix_L_Block(input_par):

    cdef double h2
    cdef complex factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"]*2.0
    
    Coeff_Plus_Minus, Coeff_Minus_Minus = Velocity_Gauge_Left_Coeff_Calculator(input_par)

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=5, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block > 0:
            if -1*m_block < l_block - 1:
                col_idx = index_map_box[(l_block - 1, m_block - 1)]*grid_size + grid_idx

                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Minus_Minus[l_block, m_block]/h2)

                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Minus[l_block, m_block] * -1*l_block/grid[grid_idx])
                
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Minus_Minus[l_block, m_block]/h2)
                

        if l_block < input_par["l_max"]:
            col_idx = index_map_box[(l_block + 1, m_block - 1)]*grid_size + grid_idx            
          
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Plus_Minus[l_block, m_block]/h2)

            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Minus[l_block, m_block]*(l_block + 1)/grid[grid_idx])
            
            #if grid_idx < grid_size - 1:
             #   Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Plus_Minus[l_block, m_block]/h2)
            
    
    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
    return Velocity_Gauge_Int_Hamiltonian 

def Velocity_Gauge_Right_Circular_Matrix_Upper_L_Block(input_par):
    cdef double h2
    cdef complex factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"]*2.0

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 

    Coeff_Plus_Plus, Coeff_Minus_Plus = Velocity_Gauge_Rigth_Coeff_Calculator(input_par)
    
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block > 0:
            if m_block < l_block - 1:
                col_idx = index_map_box[(l_block - 1, m_block + 1)]*grid_size + grid_idx                

                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Minus_Plus[l_block, m_block]/h2)

                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Plus[l_block, m_block] * -1*l_block/grid[grid_idx])
                
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Minus_Plus[l_block, m_block]/h2)
                            
    
    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
   
    return Velocity_Gauge_Int_Hamiltonian 
def Velocity_Gauge_Right_Circular_Matrix_Lower_L_Block(input_par):
    cdef double h2
    cdef complex factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"]*2.0

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 

    Coeff_Plus_Plus, Coeff_Minus_Plus = Velocity_Gauge_Rigth_Coeff_Calculator(input_par)
    
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block < input_par["l_max"]:
            col_idx = index_map_box[(l_block + 1, m_block + 1)]*grid_size + grid_idx           
         
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Plus_Plus[l_block, m_block]/h2)

            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Plus[l_block, m_block] *(l_block + 1)/grid[grid_idx])

            if grid_idx < grid_size- 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Plus_Plus[l_block, m_block]/h2)
                            
    
    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
    
    return Velocity_Gauge_Int_Hamiltonian 

def Velocity_Gauge_Left_Circular_Matrix_Upper_L_Block(input_par):
    cdef double h2
    cdef complex factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"]*2.0
    
    Coeff_Plus_Minus, Coeff_Minus_Minus = Velocity_Gauge_Left_Coeff_Calculator(input_par)

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block > 0:
            if -1*m_block < l_block - 1:
                col_idx = index_map_box[(l_block - 1, m_block - 1)]*grid_size + grid_idx

                if grid_idx >=  1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Minus_Minus[l_block, m_block]/h2)

                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Minus_Minus[l_block, m_block] * -1*l_block/grid[grid_idx])
                
                if grid_idx < grid_size - 1:
                    Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Minus_Minus[l_block, m_block]/h2)
                
    
    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
    return Velocity_Gauge_Int_Hamiltonian 

def Velocity_Gauge_Left_Circular_Matrix_Lower_L_Block(input_par):
    cdef double h2
    cdef complex factor
    cdef int l_block, m_block, grid_idx, col_idx, grid_size

    index_map_l_m, index_map_box = Mod.Index_Map_L_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size 
    matrix_size = grid_size * len(index_map_l_m)
    h2 = input_par["grid_spacing"]*2.0
    
    Coeff_Plus_Minus, Coeff_Minus_Minus = Velocity_Gauge_Left_Coeff_Calculator(input_par)

    Velocity_Gauge_Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Velocity_Gauge_Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid_size)][0]
        m_block = index_map_l_m[floor(i/grid_size)][1]
        grid_idx = i % grid_size

        if l_block < input_par["l_max"]:
            col_idx = index_map_box[(l_block + 1, m_block - 1)]*grid_size + grid_idx            
          
            if grid_idx >=  1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx - 1, -1*Coeff_Plus_Minus[l_block, m_block]/h2)

            Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx, Coeff_Plus_Minus[l_block, m_block]*(l_block + 1)/grid[grid_idx])
            
            if grid_idx < grid_size - 1:
                Velocity_Gauge_Int_Hamiltonian.setValue(i, col_idx + 1, Coeff_Plus_Minus[l_block, m_block]/h2)
                
    
    Velocity_Gauge_Int_Hamiltonian.assemblyBegin()
    Velocity_Gauge_Int_Hamiltonian.assemblyEnd()
    return Velocity_Gauge_Int_Hamiltonian 