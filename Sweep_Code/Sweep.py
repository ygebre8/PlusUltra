import numpy as np 
import h5py
import json 
import sys
import os


""" This is the function that will change the parameters in the input files. """
def Input_File_Editor(s, input_file="input.json"):
    
    """ This opens the input file and loads it into input_par """
    with open(input_file) as input_file:
            input_par = json.load(input_file)
    
    """ Here I am changing the beta parameter """
    number_of_lasers = len(input_par["laser"]["pulses"])
    for j in range(number_of_lasers):
        input_par["laser"]["pulses"][j]["beta"] = float(s)

    """ After making changes to the input file, I save the changes here """
    with open('input.json', 'w') as file:
        json.dump(input_par, file, separators=(','+'\n', ': '))

if __name__=="__main__":
   
    """ This checks to see if there exists a directory Sweep and creates it if it does not """
    if not os.path.exists('Sweep'):
        os.makedirs('Sweep')
   
    os.chdir("Sweep") ## This moves you into the Sweep directory; notice the chdir at the end
    
    sweep_range = np.arange(-4,4.2,0.2) ## creates a range of beta parameters that I want to sweep over

    """ here we have two variables for the for loop, i that just gors from 0 to the length of the sweep rnage and s that holds the variables of the sweep"""
    for i, s in enumerate(sweep_range): 
       
       """ Here I create directories for the jobs calles Job_ and a number next to it that is indexed by i"""
        Job_Name = "Job_" + str(i) 
        if not os.path.exists(Job_Name):
            os.makedirs(Job_Name)

        os.chdir(Job_Name) ## move to the Job_Name directory, here Nmae is 0,1,2 ...
        os.system('cp ../../Sample_Job/input.json .') ## here i am copying the input file from the Sameple directory and moving into the Job_Name directory that I am currently in
        os.system('cp ../../Sample_Job/Helium.h5 .') ## same for helium
        
        print(round(s,5))

        Input_File_Editor(round(s,10), input_file="input.json") ## Change the input file here, for this case just change the value of beta to the vairiable s.
        
        os.system('sbatch ~/Research/Submitscript/submit_job.sh') ## submit the job
        os.chdir('..') ## move up in directory so now i can create repeat this process for another job directory
      


        