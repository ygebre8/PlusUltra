import numpy as np 
import h5py
import json 
import sys
import os

def Input_File_Editor(s, input_file="input.json"):
    with open(input_file) as input_file:
            input_par = json.load(input_file)
    
    number_of_lasers = len(input_par["laser"]["pulses"])
    for j in range(number_of_lasers):
        input_par["laser"]["pulses"][j]["beta"] = float(s)

    with open('input.json', 'w') as file:
        json.dump(input_par, file, separators=(','+'\n', ': '))

if __name__=="__main__":
   
    if not os.path.exists('Sweep'):
        os.makedirs('Sweep')
    os.chdir("Sweep")
    
    sweep_range = np.arange(-4,4.2,0.2)

    for i, s in enumerate(sweep_range):
        Job_Name = "Job_" + str(i) 
        if not os.path.exists(Job_Name):
            os.makedirs(Job_Name)

        os.chdir(Job_Name)
        os.system('cp ../../Sample_Job/input.json .')
        os.system('cp ../../Sample_Job/Helium.h5 .')
        
        print(round(s,5))

        Input_File_Editor(round(s,10), input_file="input.json")
        
        os.system('sbatch ~/Research/Submitscript/submit_job.sh')
        os.chdir('..')
      


        