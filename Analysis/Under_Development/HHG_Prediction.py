import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def Counter_Pred(n, p_max, w=1):
    for p in range(0, p_max): 
        plt.vlines((n+1)*p*w - w, ymin=0, ymax=1.5, color = 'red', linewidth = 2.5)
        plt.vlines((n+1)*p*w + w, ymin=0, ymax=1.5, color = 'blue', linewidth = 2.5)
        
    plt.ylim(0, 2)  
    plt.xlim(0, (n+1)*p*w + w - .5)
    plt.xticks(range(1, (n+1)*p*w + w, 2))
    plt.title("Counter-rotating bicircular pulse HHG prediction")
    plt.show()


def Co_Pred(n, p_max, w=1):
    for p in range(0, p_max): 
        plt.vlines((n-1)*p*w - (n-2)*w, ymin=0, ymax=1.5, color = 'red', linewidth = 2.5)
        plt.vlines((n-1)*p*w + (n-2)*w + 0.1, ymin=0, ymax=1.5, color = 'blue', linewidth = 2.5)
        print((n-1)*p*w - (n-2)*w)
        print((n-1)*p*w + (n-2)*w)
    plt.ylim(0, 2)  
    # plt.xlim(0, (n-1)*p*w + (n-2)*w - .5)
    # plt.xticks(range(1, (n-1)*p*w + (n-2)*w, 2))
    plt.title("Co-rotating bicircular pulse HHG prediction")
    plt.show()


if __name__=="__main__":
    Co_Pred(1, 10)

    # for p in range(1, 10):
    #     # print((2*p - 1)) 
    #     print((2*p + 1)) 
    