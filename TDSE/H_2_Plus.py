if True:
    import numpy as np
    import sys
    import numpy as np
    from numpy import pi
    from math import floor
    from sympy.physics.wigner import gaunt, wigner_3j
    import json


l_values = np.arange(0, 20)
size = len(l_values)
l_values.astype(int)

Pot = np.zeros((size, size))

m = 0
for l in l_values:
    for l_prime in l_values:
        wig = 0.0
        for lamda in range(0, 30, 2):
            wig += float(wigner_3j(l_prime,lamda,l,-m,0,m)) * float(wigner_3j(l_prime,lamda,l,0,0,0))
           
        if wig > 10e-4:
            print(l, l_prime, wig)