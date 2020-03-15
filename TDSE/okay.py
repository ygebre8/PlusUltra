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