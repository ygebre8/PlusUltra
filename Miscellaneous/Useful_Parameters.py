
def Pulse_Length(omega, lamda):
    if omega!=None:
        lamda = 800*0.057/omega
        freq = 299792458 / (lamda*1e-9)
        period = 1e15/freq
        return period
    if omega==None and lamda!=None:
        freq = 299792458 / (lamda*1e-9)
        period = 1e15/freq
        return period


def HHG_Cut_Off(omega, intensity, bond_energy):
    intensity = intensity/3.51e16
    Up = intensity/(4*omega**2)
    cut_off = bond_energy + 3.17*Up
    cut_off = cut_off / omega
    return cut_off

if __name__=="__main__":
    omega = 0.057
    intensity = 3.5e14
    bond_energy = 0.6712407
    # period = Pulse_Length(0.1713, None)   
    cut_off = HHG_Cut_Off(omega, intensity, bond_energy)
    print(cut_off)