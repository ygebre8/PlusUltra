import numpy as np 
import h5py
import json 
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=="__main__":

    os.chdir("Sweep_Main")
    

    sweep_range = np.arange(-4,4.1,0.1)
  
    pop_array = [] 
    for i, s in enumerate(sweep_range):

        Job_Name = "Job_" + str(i)
        os.chdir(Job_Name)

        # os.system('rm pop.txt')
        print(Job_Name)
        # os.system('python ~/Research/PlusUltra/Analysis/Population_Calculator.py >> pop.txt')
        file = open("pop.txt", "r")
        
        pop = file.read()
        print(s, pop)
        pop_array.append(round(float(pop),5))

        
        os.chdir('..')

    plt.plot(sweep_range, pop_array)
    plt.savefig("ion.png")
    