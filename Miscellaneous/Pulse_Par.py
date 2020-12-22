
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


if __name__=="__main__":
    period = Pulse_Length(0.1713, None)   
    print(period)