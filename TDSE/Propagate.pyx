if True:
    import numpy as np
    import sys
    import time as time_mod
    import Laser_Pulse as LP
    import Module as Mod
    import Interaction as Int
    import Field_Free as FF
    import Dipole_Acceleration_Matrix as DA


if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    petsc4py.init(comm=PETSc.COMM_WORLD)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()



def Build_Psi(psi_inital):

    psi = PETSc.Vec().createMPI(np.size(psi_inital), comm=PETSc.COMM_WORLD)
    istart, iend = psi.getOwnershipRange()
    for i  in range(istart, iend):
        psi.setValue(i, psi_inital[i])

    psi.assemblyBegin()
    psi.assemblyEnd()
    return psi

def Crank_Nicolson_Time_Propagator(input_par, psi_inital):

    laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse = LP.Build_Laser_Pulse(input_par)    
    
    FF_Ham = eval("FF.Build_FF_Hamiltonian_" + input_par["order"] +"_Order_" + input_par["block_type"] + "(input_par)")
    FF_Ham.scale(-1.0j * input_par["time_spacing"] * 0.5)

    build_status = Mod.Matrix_Build_Status(input)
    matrix_size = FF_Ham.getSize()[0]
    if input_par["gauge"] == "Length":
        Full_Ham = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=6, comm=PETSc.COMM_WORLD)
    if input_par["gauge"] == "Velocity":
        Full_Ham = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=12, comm=PETSc.COMM_WORLD)

    FF_Ham.copy(Full_Ham, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)


    if elliptical_pulse == False:
        if rank == 0:
            print("Building Matrices for Linear Pulse \n ")
        
        if np.dot(total_polarization, np.array([1,0,0])) != 0:   
           
            Int_Ham_X = eval("Int." + input_par["gauge"] + "_Gauge_X_Matrix_" + input_par["block_type"] + "(input_par)")
            Int_Ham_X.scale(-1.0j * input_par["time_spacing"] * 0.5)
            Full_Ham.axpy(0.0, Int_Ham_X, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            build_status["Int_Mat_X_Stat"] = True
            if rank == 0:
                print("Built the "+  str(input_par["gauge"]) + " Gauge X Interaction Matrix \n")

            if input_par["HHG"] == 1:
                Dip_Acc_Mat_X = eval("DA.Dipole_Acceleration_X_Matrix_" + input_par["block_type"] + "(input_par)")
                Dip_Acc_X = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Acc_Mat_X_Stat"] = True
                if rank == 0:
                    print("Built the X Dipole_Acceleration Matrix \n")
            
            if input_par["Dipole"] == 1:
                    Dip_Mat_X =  Int.Length_Gauge_X_Matrix_L_Block(input_par)
                    Dip_X = np.zeros(len(laser_time), dtype=complex)
                    build_status["Dip_Mat_X_Stat"] = True
                    if rank == 0:
                        print("Built the X Dipole Matrix \n")

        if np.dot(total_polarization, np.array([0,1,0])) != 0 : 

            Int_Ham_Y = eval("Int." + input_par["gauge"] + "_Gauge_Y_Matrix_" + input_par["block_type"] + "(input_par)")
            Int_Ham_Y.scale(-1.0j * input_par["time_spacing"] * 0.5)
            Full_Ham.axpy(0.0, Int_Ham_Y, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            build_status["Int_Mat_Y_Stat"] = True
            if rank == 0:
                 print("Built the "+  str(input_par["gauge"]) + " Gauge Y Interaction Matrix \n")

            if input_par["HHG"] == 1:
                Dip_Acc_Mat_Y = eval("DA.Dipole_Acceleration_Y_Matrix_" + input_par["block_type"] + "(input_par)")
                Dip_Acc_Y = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Acc_Mat_Y_Stat"] = True
                if rank == 0:
                    print("Built the Y Dipole_Acceleration Matrix \n")
        
            if input_par["Dipole"] == 1:
                    Dip_Mat_Y =  Int.Length_Gauge_Y_Matrix_L_Block(input_par)
                    Dip_Y = np.zeros(len(laser_time), dtype=complex)
                    build_status["Dip_Mat_Y_Stat"] = True
                    if rank == 0:
                        print("Built the Y Dipole Matrix \n")
        
        
        if np.dot(total_polarization, np.array([0,0,1])) != 0: 
            Int_Ham_Z = eval("Int." + input_par["gauge"] + "_Gauge_Z_Matrix_" + input_par["block_type"] + "(input_par)")
            Int_Ham_Z.scale(-1.0j * input_par["time_spacing"] * 0.5)
            Full_Ham.axpy(0.0, Int_Ham_Z, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            build_status["Int_Mat_Z_Stat"] = True
            if rank == 0:
                 print("Built the "+  str(input_par["gauge"]) + " Gauge Z Interaction Matrix \n")
            
            if input_par["HHG"] == 1:
                Dip_Acc_Mat_Z = eval("DA.Dipole_Acceleration_Z_Matrix_" + input_par["block_type"] + "(input_par)")
                Dip_Acc_Z = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Acc_Mat_Z_Stat"] = True
                if rank == 0:
                    print("Built the Z Dipole_Acceleration Matrix \n")
            
            if input_par["Dipole"] == 1:
                    Dip_Mat_Z =  Int.Length_Gauge_Z_Matrix_L_Block(input_par)
                    Dip_Z = np.zeros(len(laser_time), dtype=complex)
                    build_status["Dip_Mat_Z_Stat"] = True
                    if rank == 0:
                        print("Built the Z Dipole Matrix \n")

    if elliptical_pulse == True:
        
        if rank == 0:
            print("Building Matrices for Elliptical Pulse \n ")
            
        Int_Ham_Right = eval("Int." + input_par["gauge"] + "_Gauge_Right_Circular_Matrix_" + input_par["block_type"] + "(input_par)")        
        Int_Ham_Right.scale(-1.0j * input_par["time_spacing"] * 0.5)
        Full_Ham.axpy(0.0, Int_Ham_Right, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        build_status["Int_Mat_Right_Stat"]  = True

        Int_Ham_Left = eval("Int." + input_par["gauge"] + "_Gauge_Left_Circular_Matrix_" + input_par["block_type"] + "(input_par)")
        Int_Ham_Left.scale(-1.0j * input_par["time_spacing"] * 0.5)
        Full_Ham.axpy(0.0, Int_Ham_Left, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        build_status["Int_Mat_Left_Stat"]  = True

        if rank == 0:
                print("Built the "+  str(input_par["gauge"]) + " Gauge Circular Interaction Matrix \n")
                

        if input_par["HHG"] == 1:
                Dip_Acc_Mat_X = eval("DA.Dipole_Acceleration_X_Matrix_" + input_par["block_type"] + "(input_par)")
                Dip_Acc_X = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Acc_Mat_X_Stat"] = True
                if rank == 0:
                    print("Built the X Dipole_Acceleration Matrix \n")

                Dip_Acc_Mat_Y = eval("DA.Dipole_Acceleration_Y_Matrix_" + input_par["block_type"] + "(input_par)")
                Dip_Acc_Y = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Acc_Mat_Y_Stat"] = True
                if rank == 0:
                    print("Built the Y Dipole_Acceleration Matrix \n")

        if input_par["Dipole"] == 1:
                Dip_Mat_X =  Int.Length_Gauge_X_Matrix_L_Block(input_par)
                Dip_X = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Mat_X_Stat"] = True
                if rank == 0:
                    print("Built the X Dipole Matrix \n")

                Dip_Mat_Y =  Int.Length_Gauge_Y_Matrix_L_Block(input_par)
                Dip_Y = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Mat_Y_Stat"] = True
                if rank == 0:
                    print("Built the Y Dipole Matrix \n")
    

        if np.dot(total_poynting,np.array([0,0,1])) != 1: 
            Int_Ham_Z = eval("Int." + input_par["gauge"] + "_Gauge_Z_Matrix_" + input_par["block_type"] + "(input_par)")
            Int_Ham_Z.scale(-1.0j * input_par["time_spacing"] * 0.5)
            Full_Ham.axpy(0.0, Int_Ham_Z, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            build_status["Int_Mat_Z_Stat"] = True
            if rank == 0:
                 print("Built the "+  str(input_par["gauge"]) + " Gauge Z Interaction Matrix \n")
            
            if input_par["HHG"] == 1:
                Dip_Acc_Mat_Z = eval("DA.Dipole_Acceleration_Z_Matrix_" + input_par["block_type"] + "(input_par)")
                Dip_Acc_Z = np.zeros(len(laser_time), dtype=complex)
                build_status["Dip_Acc_Mat_Z_Stat"]
                if rank == 0:
                    print("Built the Z Dipole_Acceleration Matrix \n")
            
            if input_par["Dipole"] == 1:
                    Dip_Mat_Z =  Int.Length_Gauge_Z_Matrix_L_Block(input_par)
                    Dip_Z = np.zeros(len(laser_time), dtype=complex)
                    build_status["Dip_Mat_Z_Stat"] = True
                    if rank == 0:
                        print("Built the Z Dipole Matrix \n")
            
    
    
    
    Full_Ham.assemblyBegin()
    Full_Ham.assemblyEnd()
    Full_Ham_Left = Full_Ham.duplicate()


    Psi = Build_Psi(psi_inital)
    Psi_Right = Psi.duplicate()
    Psi_Dipole = Psi.duplicate()     

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOptionsPrefix("prop_")

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(input_par["TDSE_File"], mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    def Build_Time_Dep_Hamiltonian(i, t):
        FF_Ham.copy(Full_Ham, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

        if build_status["Int_Mat_X_Stat"]: 
            Full_Ham.axpy(laser_pulse['x'][i], Int_Ham_X, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            
            if build_status["Dip_Acc_Mat_X_Stat"]:
                Dip_Acc_Mat_X.mult(Psi, Psi_Dipole)
                Dip_Acc_X[i] = Psi_Dipole.dot(Psi)
            
            if build_status["Dip_Mat_X_Stat"]:
                Dip_Mat_X.mult(Psi, Psi_Dipole)
                Dip_X[i] = Psi_Dipole.dot(Psi)

        if build_status["Int_Mat_Y_Stat"]: 
            Full_Ham.axpy(laser_pulse['y'][i], Int_Ham_Y, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            
            if build_status["Dip_Acc_Mat_Y_Stat"]:
                Dip_Acc_Mat_Y.mult(Psi, Psi_Dipole)
                Dip_Acc_Y[i] = Psi_Dipole.dot(Psi)

            if build_status["Dip_Mat_Y_Stat"]:
                Dip_Mat_Y.mult(Psi, Psi_Dipole)
                Dip_Y[i] = Psi_Dipole.dot(Psi)

        
        if build_status["Int_Mat_Z_Stat"]:  
            Full_Ham.axpy(laser_pulse['z'][i], Int_Ham_Z, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
           
            if build_status["Dip_Acc_Mat_Z_Stat"]:
                Dip_Acc_Mat_Z.mult(Psi, Psi_Dipole)
                Dip_Acc_Z[i] = Psi_Dipole.dot(Psi)

            if build_status["Dip_Mat_Z_Stat"]:
                Dip_Mat_Z.mult(Psi, Psi_Dipole)
                Dip_Z[i] = Psi_Dipole.dot(Psi)
   

        if build_status["Int_Mat_Right_Stat"]:
            Full_Ham.axpy(laser_pulse['Right'][i], Int_Ham_Right, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
           
            if build_status["Dip_Acc_Mat_X_Stat"]:
                Dip_Acc_Mat_X.mult(Psi, Psi_Dipole)
                Dip_Acc_X[i] = Psi_Dipole.dot(Psi)
            
            if build_status["Dip_Mat_X_Stat"]:
                Dip_Mat_X.mult(Psi, Psi_Dipole)
                Dip_X[i] = Psi_Dipole.dot(Psi)
                
                

        if build_status["Int_Mat_Left_Stat"]:
            Full_Ham.axpy(laser_pulse['Left'][i], Int_Ham_Left, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            
            if build_status["Dip_Acc_Mat_Y_Stat"]:
                Dip_Acc_Mat_Y.mult(Psi, Psi_Dipole)
                Dip_Acc_Y[i] = Psi_Dipole.dot(Psi)

            if build_status["Dip_Mat_Y_Stat"]:
                Dip_Mat_Y.mult(Psi, Psi_Dipole)
                Dip_Y[i] = Psi_Dipole.dot(Psi)

                
    if rank == 0:
            print("Starting time propagation")
            start_time = time_mod.time()

    interval = int(len(laser_time)/input_par["psi_write_frequency"])
    psi_save_array = np.arange(interval , interval + len(laser_time), interval)
    save_count = 0

    for i, t in enumerate(laser_time):
        
        Build_Time_Dep_Hamiltonian(i, t)
        Full_Ham.copy(Full_Ham_Left, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        
        Full_Ham_Left.scale(-1.0)
        Full_Ham_Left.shift(1.0)
        Full_Ham.shift(1.0)

        ksp.setOperators(Full_Ham_Left, Full_Ham_Left)
        ksp.setTolerances(1.e-12, PETSc.DEFAULT, PETSc.DEFAULT, PETSc.DEFAULT)
        ksp.setFromOptions()

        Full_Ham.mult(Psi, Psi_Right)
        ksp.solve(Psi_Right, Psi)
        
        if rank == 0:
            print(i, len(laser_time)-1)
    
        if i + 1 in psi_save_array:
            if rank == 0:
                print("Saving Wavefunction at " + str(i))
    
            Psi_name = "Psi" + str(save_count)
            save_count += 1
            Psi.setName(Psi_name) 
            ViewHDF5.view(Psi)
            
    Psi.setName("Psi_Final") 
    ViewHDF5.view(Psi)
    ViewHDF5.destroy()
    
    if rank == 0:
            print("Finished time propagation")
            print(str(round((time_mod.time() - start_time) / 60, 3)) + " minutes" ) 
    
    if build_status["Dip_Acc_Mat_X_Stat"]:
        np.savetxt("Dip_Acc_X.txt", Dip_Acc_X.view(float))
        
    if build_status["Dip_Acc_Mat_Y_Stat"]:
        np.savetxt("Dip_Acc_Y.txt", Dip_Acc_Y.view(float))
        
    if build_status["Dip_Acc_Mat_Z_Stat"]:
        np.savetxt("Dip_Acc_Z.txt", Dip_Acc_Z.view(float))
        
    if build_status["Dip_Mat_X_Stat"]:
        np.savetxt("Dip_X.txt", Dip_X.view(float))

    if build_status["Dip_Mat_Y_Stat"]:
        np.savetxt("Dip_Y.txt", Dip_Y.view(float))
        
    if build_status["Dip_Mat_Z_Stat"]:
        np.savetxt("Dip_Z.txt", Dip_Z.view(float))

    np.savetxt("time.txt", laser_time)

    return None
