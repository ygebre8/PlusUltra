if True:
    import numpy as np
    import sys
    from numpy import pi
    from math import floor
    from sympy.physics.wigner import gaunt, wigner_3j
    import json


l_values = np.arange(0, 150)
l_values.astype(int)

wigner_3j_dict = {}



for l in l_values:
    for m in np.arange(-1* l, l + 1):

        l_prime = l+1

        m_prime = m
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,0))] =  float(wigner_3j(l,1,l_prime,-1*m,0,m_prime))


        m_prime = m+1
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))] =  float(wigner_3j(l,1,l_prime,-1*m,1,m_prime))
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] = float(wigner_3j(l,1,l_prime,-1*m,-1,m_prime))
        
        m_prime = m-1
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))] =  float(wigner_3j(l,1,l_prime,-1*m,1,m_prime))
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] = float(wigner_3j(l,1,l_prime,-1*m,-1,m_prime))

        wigner_3j_dict[str((l,0,l_prime,0,1,0))] =  float(wigner_3j(l,1,l_prime,0,0,0))


        l_prime = l-1

        m_prime = m
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,0))] =  float(wigner_3j(l,1,l_prime,-1*m,0,m_prime))

        m_prime = m+1
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))] =  float(wigner_3j(l,1,l_prime,-1*m,1,m_prime))
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] = float(wigner_3j(l,1,l_prime,-1*m,-1,m_prime))
        
        m_prime = m-1
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))] =  float(wigner_3j(l,1,l_prime,-1*m,1,m_prime))
        wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] = float(wigner_3j(l,1,l_prime,-1*m,-1,m_prime))

        wigner_3j_dict[str((l,0,l_prime,0,1,0))] =  float(wigner_3j(l,1,l_prime,0,0,0))
        
with open("/home/becker/yoge8051/Research/YOSHI/TDSE/wigner_3j.json", 'w') as file:
    json.dump(wigner_3j_dict, file, separators=(','+'\n', ': '))


# with open("wigner_3j.json") as file:
#     wigner_3j_dict = json.load(file)


# print(wigner_3j_dict.keys())