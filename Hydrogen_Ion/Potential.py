import numpy as np
from sympy.physics.wigner import gaunt, wigner_3j
import json
import H2_Module as Mod 
import matplotlib.pyplot as plt

def Nucleus_Electron_Interaction(grid, l, l_prime, m, R_o):

    R_o = R_o / 2.0
    R_o_idx =  np.nonzero(grid > R_o)[0][0]
    
    grid_low = grid[:R_o_idx]
    grid_high = grid[R_o_idx:]

    potential_low = np.zeros(len(grid_low))
    potential_high = np.zeros(len(grid_high))


    for lamda in range(0, l + l_prime + 40, 2):

        coef = float(wigner_3j(l,lamda,l_prime,-m,0,m)) * float(wigner_3j(l,lamda,l_prime,0,0,0))    
        potential_low +=  (np.power(grid_low, lamda) / float(pow(R_o, lamda+1))) * coef
        potential_high +=  (float(pow(R_o, lamda)) / np.power(grid_high, lamda + 1)) * coef

    potential = -2.0 * np.sqrt((2*l+1)*(2*l_prime+1))* pow(-1,m)*np.concatenate((potential_low, potential_high))
    potential = list(potential)

    return potential

    
def Potential(input_par):
    pot_dict = {}

    for m in range(input_par["m_max"] + 1):
        for l in range(input_par["l_max"] + 1):
            # print(m, l)
            if l % 2 == 0:
                l_prime_list = range(0, input_par["l_max"] + 1, 2)
            else:
                l_prime_list = range(1, input_par["l_max"] + 1, 2)

            for l_prime in l_prime_list:
                pot_dict[str((m, l, l_prime))] = Nucleus_Electron_Interaction(grid, l, l_prime, m, input_par["R_o"])

    return pot_dict

if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    
    pot_dict = Potential(input_par)
    with open("Nuclear_Electron_Int.json", 'w') as file:
        json.dump(pot_dict, file)

    # l = 10
    # l_prime = 20

    # potential =  Nucleus_Electron_Interaction(grid, 6, l_prime, 1, 2)
    # plt.plot(grid, potential)
    # potential =  Nucleus_Electron_Interaction(grid, 6, l_prime, 1, 2)
    # plt.plot(grid, potential)

    # potential =  Nucleus_Electron_Interaction(grid, 60, 2, 0, 51, 80)
    # plt.plot(grid, potential)

    # # plt.plot(-1/grid)
    # plt.xlim(0, 10)
    # # plt.show()
    # plt.savefig("pic.png")