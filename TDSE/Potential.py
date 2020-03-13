import numpy as np

def Coulomb_Eff_Potential(grid, l):
    return -1.0*np.power(grid, -1.0) + 0.5*l*(l+1)*np.power(grid, -2.0)


def SAE(grid, l):
    return -1.0*np.power(grid, -1.0) + -1.0*np.exp(-2.0329*grid)/grid  - 0.3953*np.exp(-6.1805*grid) + 0.5*l*(l+1)*np.power(grid, -2.0)

    