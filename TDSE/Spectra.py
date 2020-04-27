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

def Continuum_Wavefuncion_Reader(input_par, z = 1):
    CWF = {}
    phase = {}
    energy = {}

    Target_file =h5py.File("Continum.h5")
    
    dr = input_par["grid_spacing"]
    r = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    r_val = r[-2]

    for l in range(0, input_par["l_max"]+ 1):
        dataset_name = "Energy_" + str(l) + "_" + "0"
        energy[l] = Target_file[dataset_name]
        energy[l] = np.array(energy[l][:,0] + 1.0j*energy[l][:,1])

        dataset_name = "Psi_" + str(l) + "_" + "0"
        CWF[l] = Target_file[dataset_name]
        CWF[l] = np.array(CWF[l][:,0] + 1.0j*CWF[l][:,1])


        k = np.linalg.norm(np.sqrt(2*energy[l]))
        # k = np.linalg.norm(np.sqrt(2*0.6874690338347257))
        
        coul_wave_r = -1*np.linalg.norm(CWF[l][-2])
        dcoul_wave_r = -1*np.linalg.norm((CWF[l][-1]-CWF[l][-3])/(2*dr))

        # norm = np.sqrt(np.abs(coul_wave_r)**2+(np.abs(dcoul_wave_r)/(k+z/(k*r_val)))**2)
        norm = np.array(CWF[l]).max()
        CWF[l] /= norm
        
        coul_wave_r = -1*np.linalg.norm(CWF[l][-2])
        dcoul_wave_r = -1*np.linalg.norm((CWF[l][-1]-CWF[l][-3])/(2*dr))
        

        
            
        phase[l] = np.angle((1.j*coul_wave_r + dcoul_wave_r/(k+z/(k*r_val))) /
                     (2*k*r_val)**(1.j*z/k)) - k*r_val + l*np.pi/2

        
        print("mine", l, np.exp(-1.j*phase[l]), k, coul_wave_r, dcoul_wave_r)

    return phase, CWF, energy

def Continuum_Wavefuncion_Maker(input_par):

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5("Continum.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    energy = 0.6874690338347257

    for l in range(0, input_par["l_max"]+ 1):

        if rank == 0:
            print("Calculating the eigenstates for l = " + str(l) + "\n")
    
        potential = eval("Pot." + input_par["potential"] + "(grid, l)")
        Hamiltonian = eval("TS.Build_Hamiltonian_" + input_par["order"] + "_Order(potential, grid, input_par)")        
        
        Eigen_Value_Solver(Hamiltonian, input_par, l, energy, ViewHDF5)

        if rank == 0:
            print("Finished calculation for l = " + str(l) + "\n" , "\n")

    ViewHDF5.destroy()
    
def Eigen_Value_Solver(Hamiltonian,input_par, l, energy, Viewer):
    number_of_eigenvalues = 1
    k = np.sqrt(2*energy)

    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    EV_Solver.setOperators(Hamiltonian) ##pass the hamiltonian to the 
    # EV_Solver.setType(SLEPc.EPS.Type.JD)
    EV_Solver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    EV_Solver.setTolerances(input_par["tolerance"], PETSc.DECIDE)
    EV_Solver.setWhichEigenpairs(EV_Solver.Which.TARGET_REAL)
    EV_Solver.setTarget(energy)
    size_of_matrix = PETSc.Mat.getSize(Hamiltonian)
    dimension_size = int(size_of_matrix[0]) * 0.1
    EV_Solver.setDimensions(number_of_eigenvalues, PETSc.DECIDE, dimension_size) 
    EV_Solver.solve() ##solve the eigenvalue problem
    eigen_states = [[],[]]
   
    for i in range(number_of_eigenvalues):
        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)
        eigen_vector.setName("Psi_" + str(l) + "_" + str(i)) 

        Viewer.view(eigen_vector)
        
        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        energy.setValue(0,eigen_state)

        energy.setName("Energy_" + str(l) + "_" + str(i))
        energy.assemblyBegin()
        energy.assemblyEnd()

        Viewer.view(energy)  

def Shooting_Method(energy, l, z = 1):
    k = np.sqrt(2*energy)
    dr = input_par["grid_spacing"]
    r = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    r2 = r*r

    potential = -1.0/r 
    coul_wave = np.zeros(r.shape)
    coul_wave[0] = 1.0
    coul_wave[1] = coul_wave[0]*(dr*dr*((l*(l+1)/r2[0])+2*potential[0]+2*energy)+2)

    for idx in np.arange(2, r.shape[0]):
        coul_wave[idx] = coul_wave[idx-1]*(dr*dr*((l*(l+1)/r2[idx-1]) + 2*potential[idx-1]-2*energy)+2) - coul_wave[idx-2]



    r_val = r[-2]
    
    coul_wave_r = coul_wave[-2]
    dcoul_wave_r = (coul_wave[-1]-coul_wave[-3])/(2*dr)

    norm = np.sqrt(np.abs(coul_wave_r)**2+(np.abs(dcoul_wave_r)/(k+z/(k*r_val)))**2)
    # norm = np.array(coul_wave).max()
    coul_wave /= norm

    coul_wave_r = coul_wave[-2]
    dcoul_wave_r = (coul_wave[-1]-coul_wave[-3])/(2*dr)

    

    phase = np.angle((1.j*coul_wave_r + dcoul_wave_r/(k+z/(k*r_val))) /
                     (2*k*r_val)**(1.j*z/k)) - k*r_val + l*np.pi/2
    
    print("joel", l, np.exp(-1.j*phase), k, coul_wave_r, dcoul_wave_r)
    
    # print((l, phase))
    # phase = 0.0
    # make sure to reshape coul_wave to the right size
    return phase, coul_wave[:r.shape[0]]

def K_Sphere(input_par):
    phi = np.arange(-np.pi, np.pi, input_par["d_angle"])
    theta = np.arange(0, np.pi+input_par["d_angle"], input_par["d_angle"])
    theta, phi = np.meshgrid(theta, phi)

    Psi = Psi_Reader(input_par)
    Bound_States = Bound_Reader(input_par)
    out_going_wave = np.zeros(phi.shape, dtype=complex)
    
    phase_dic, CWF, energy =  Continuum_Wavefuncion_Reader(input_par) 

    for l in range(0, input_par["l_max"]+ 1):    
        phase = phase_dic[l]
        coul_wave = CWF[l]
        n_max = input_par["n_max"]
        m_range = min(l, input_par["m_max"])
        for m in range(-1*m_range, m_range + 1):
            for n in range(l + 1, n_max + 1):
                Psi[(l, m)] -= np.sum(Bound_States[(n,l)].conj()*Psi[(l,m)])*Bound_States[(n,l)]
        
            coef = np.exp(-1.j*phase)*1.j**l * np.sum(coul_wave.conj()*Psi[(l,m)])
            out_going_wave += coef*sph_harm(m, l, phi, theta)
    return phi, theta, out_going_wave

if __name__=="__main__":

    input_par = Mod.Input_File_Reader("input.json")
    
    

    # r = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    # phase, CWF, energy = Continuum_Wavefuncion_Reader(input_par, z = 1)
    # # for l in range(26):
    #     phase_old, coul_wave_old = Shooting_Method(np.linalg.norm(energy[l]), l, z = 1)


    # # print(phase_old, phase[l])
    # # plt.plot(r,coul_wave_old)
    # # plt.plot(r, -1*CWF[l])
    # plt.xlim(400,550)
    # plt.savefig("comp.png")


    # phi, theta, out_going_wave = K_Sphere(input_par)
    # out_going_wave = np.abs(out_going_wave)**2

    # out_going_wave = out_going_wave/out_going_wave.max()

    # X, Y, Z = out_going_wave*np.sin(theta)*np.cos(phi), out_going_wave *np.sin(theta)*np.sin(phi), out_going_wave*np.cos(theta)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # cmap = cm.get_cmap("viridis")
    # ax.plot_surface(X, Y, Z, facecolors=cmap(out_going_wave))
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # plt.savefig("PAD_New.png")


    Continuum_Wavefuncion_Maker(input_par)
    # CWF, phase =  Continuum_Wavefuncion_Reader(input_par) 
    # r = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    
        