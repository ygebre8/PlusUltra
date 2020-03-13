import numpy as np
import json 
import sys
from math import sqrt

def Length_Gauge_Z_Coeff_Calculator(input_par):
    Coeff_Upper = {}
    Coeff_Lower = {}
    cdef int l, m 
    for l in np.arange(input_par["l_max"] + 1):
        for m in np.arange(-1*l, l+1):
            Coeff_Upper[l,m] = np.sqrt((pow(l+1, 2.0) - pow(m, 2.0)) / (4*pow(l+1, 2.0) - 1))
            Coeff_Lower[l,m] = np.sqrt((pow(l, 2.0) - pow(m, 2.0)) / (4*pow(l, 2.0) - 1))

    return Coeff_Upper, Coeff_Lower

def Length_Gauge_X_Coeff_Calculator(input_par):
    Coeff_Plus_Plus = {}
    Coeff_Minus_Plus = {}
    Coeff_Plus_Minus = {}
    Coeff_Minus_Minus = {}

    with open(sys.path[0] + "/wigner_3j.json") as file:
        wigner_3j_dict = json.load(file)
    
    cdef int l, m 
    cdef double factor
    for l in np.arange(input_par["l_max"] + 1):
        for m in np.arange(-1*l, l+1):
            if l < input_par["l_max"]:
                factor = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l+3)/2)*wigner_3j_dict[str((l,0,l+1,0,1,0))]
                Coeff_Plus_Plus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l+1,m+1,1,-1))] - wigner_3j_dict[str((l,-1*m,l+1,m+1,1,1))])
                Coeff_Plus_Minus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l+1 ,m-1,1,-1))] - wigner_3j_dict[str((l,-1*m,l+1 ,m-1,1,1))])
            if l > 0:
                factor = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l-1)/2)*wigner_3j_dict[str((l,0,l-1,0,1,0))]
                if m < l - 1:
                    Coeff_Minus_Plus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l-1,m+1,1,-1))] - wigner_3j_dict[str((l,-1*m,l-1,m+1,1,1))])
                if -1*m < l - 1:
                    Coeff_Minus_Minus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l-1,m-1,1,-1))] - wigner_3j_dict[str((l,-1*m,l-1,m-1,1,1))])    

    return Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus

def Length_Gauge_Y_Coeff_Calculator(input_par):
    Coeff_Plus_Plus = {}
    Coeff_Minus_Plus = {}
    Coeff_Plus_Minus = {}
    Coeff_Minus_Minus = {}

    with open(sys.path[0] + "/wigner_3j.json") as file:
        wigner_3j_dict = json.load(file)
    
    cdef int l, m 
    cdef complex factor
    for l in np.arange(input_par["l_max"] + 1):
        for m in np.arange(-1*l, l+1):
            if l < input_par["l_max"]:
                factor = 1.0j*pow(-1.0, m)*np.sqrt((2*l+1)*(2*l+3)/2)*wigner_3j_dict[str((l,0,l+1,0,1,0))]
                Coeff_Plus_Plus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l+1,m+1,1,-1))] + wigner_3j_dict[str((l,-1*m,l+1,m+1,1,1))])
                Coeff_Plus_Minus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l+1,m-1,1,-1))] + wigner_3j_dict[str((l,-1*m,l+1,m-1,1,1))])
            if l > 0:
                factor = 1.0j*pow(-1.0, m)*np.sqrt((2*l+1)*(2*l-1)/2)*wigner_3j_dict[str((l,0,l-1,0,1,0))]
                if m < l - 1:
                    Coeff_Minus_Plus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l-1,m+1,1,-1))] + wigner_3j_dict[str((l,-1*m,l-1,m+1,1,1))])
                if -1*m < l - 1:
                    Coeff_Minus_Minus[l,m] = factor*(wigner_3j_dict[str((l,-1*m,l-1,m-1,1,-1))] + wigner_3j_dict[str((l,-1*m,l-1,m-1,1,1))]) 

    return Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus

def Length_Gauge_Right_Coeff_Calculator(input_par):
        Coeff_Plus_Plus = {}
        Coeff_Minus_Plus = {}
        
        with open(sys.path[0] + "/wigner_3j.json") as file:
            wigner_3j_dict = json.load(file)

        cdef int l, m 
        cdef double factor
        for l in np.arange(input_par["l_max"] + 1):
            for m in np.arange(-1*l, l+1):

                if l < input_par["l_max"]:
                    factor = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l+3)/2)*wigner_3j_dict[str((l,0,l+1,0,1,0))]
                    x_term = (wigner_3j_dict[str((l,-1*m,l+1,m+1,1,-1))] - wigner_3j_dict[str((l,-1*m,l+1,m+1,1,1))])
                    y_term = (wigner_3j_dict[str((l,-1*m,l+1,m+1,1,-1))] + wigner_3j_dict[str((l,-1*m,l+1,m+1,1,1))])
                    Coeff_Plus_Plus[l,m] = factor*(x_term + y_term)

                if l > 0:
                    if m < l - 1:
                        factor = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l-1)/2)*wigner_3j_dict[str((l,0,l-1,0,1,0))]
                        x_term = (wigner_3j_dict[str((l,-1*m,l-1,m+1,1,-1))] - wigner_3j_dict[str((l,-1*m,l-1,m+1,1,1))])
                        y_term = (wigner_3j_dict[str((l,-1*m,l-1,m+1,1,-1))] + wigner_3j_dict[str((l,-1*m,l-1,m+1,1,1))])
                        Coeff_Minus_Plus[l,m] = factor*(x_term + y_term)   

        return Coeff_Plus_Plus, Coeff_Minus_Plus

def Length_Gauge_Left_Coeff_Calculator(input_par):
        Coeff_Plus_Minus = {}
        Coeff_Minus_Minus = {}
        
        with open(sys.path[0] + "/wigner_3j.json") as file:
            wigner_3j_dict = json.load(file)

        cdef int l, m 
        cdef double factor
        for l in np.arange(input_par["l_max"] + 1):
            for m in np.arange(-1*l, l+1):

                if l < input_par["l_max"]:
                    factor = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l+3)/2)*wigner_3j_dict[str((l,0,l+1,0,1,0))]
                    x_term = (wigner_3j_dict[str((l,-1*m,l+1 ,m-1,1,-1))] - wigner_3j_dict[str((l,-1*m,l+1 ,m-1,1,1))])
                    y_term = (wigner_3j_dict[str((l,-1*m,l+1,m-1,1,-1))] + wigner_3j_dict[str((l,-1*m,l+1,m-1,1,1))])
                    Coeff_Plus_Minus[l,m] = factor*(x_term - y_term)

                if l > 0:
                    if -1*m < l - 1:
                        factor = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l-1)/2)*wigner_3j_dict[str((l,0,l-1,0,1,0))]
                        x_term = (wigner_3j_dict[str((l,-1*m,l-1,m-1,1,-1))] - wigner_3j_dict[str((l,-1*m,l-1,m-1,1,1))])
                        y_term = (wigner_3j_dict[str((l,-1*m,l-1,m-1,1,-1))] + wigner_3j_dict[str((l,-1*m,l-1,m-1,1,1))])
                        Coeff_Minus_Minus[l,m] = factor*(x_term  - y_term)   

        return Coeff_Plus_Minus, Coeff_Minus_Minus

def Velocity_Gauge_X_Coeff_Calculator(input_par):
    Coeff_Plus_Plus = {}
    Coeff_Minus_Plus = {}
    Coeff_Plus_Minus = {}
    Coeff_Minus_Minus = {}
    cdef int l, m 
    cdef complex factor
    for l in np.arange(input_par["l_max"] + 1):
        for m in np.arange(-1*l, l+1):
            if l < input_par["l_max"]:
                factor = 0.5j*sqrt((l+m+1)*(l+m+2)/(4*pow(l+1, 2.0) - 1))
                Coeff_Plus_Plus[l,m] = factor
                factor = 0.5j*sqrt((l-m+1)*(l-m+2)/(4*pow(l+1, 2.0) - 1))
                Coeff_Plus_Minus[l,m] = factor
            if l > 0:
                if -1*m < l-1:
                    factor = 0.5j*sqrt((l+m)*(l+m-1)/(4*pow(l, 2.0) - 1)) 
                    Coeff_Minus_Minus[l,m] = factor
                if m < l-1:
                    factor = 0.5j*sqrt((l-m)*(l-m-1)/(4*pow(l, 2.0) - 1))
                    Coeff_Minus_Plus[l,m] = factor

    return Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus

def Velocity_Gauge_Y_Coeff_Calculator(input_par):
    Coeff_Plus_Plus = {}
    Coeff_Minus_Plus = {}
    Coeff_Plus_Minus = {}
    Coeff_Minus_Minus = {}
    cdef int l, m 
    cdef complex factor
    for l in np.arange(input_par["l_max"] + 1):
        for m in np.arange(-1*l, l+1):
            if l < input_par["l_max"]:
                factor = -0.5*sqrt((l-m+1)*(l-m+2)/(4*pow(l+1, 2.0) - 1))
                Coeff_Plus_Minus[l,m] = factor
                factor = 0.5*sqrt((l+m+1)*(l+m+2)/(4*pow(l+1, 2.0) - 1))
                Coeff_Plus_Plus[l,m] = factor
            if l > 0:
                if -1*m < l-1:
                    factor = -0.5*sqrt((l+m)*(l+m-1)/(4*pow(l, 2.0) - 1)) 
                    Coeff_Minus_Minus[l,m] = factor
                if m < l-1:
                    factor = 0.5*sqrt((l-m)*(l-m-1)/(4*pow(l, 2.0) - 1))
                    Coeff_Minus_Plus[l,m] =factor
            
             

    return Coeff_Plus_Plus, Coeff_Minus_Plus, Coeff_Plus_Minus, Coeff_Minus_Minus