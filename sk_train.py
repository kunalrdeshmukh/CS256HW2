import sys
import math
import numpy as np
from SK import SKAlgo

KERNEL_TYPE = 'P'  # P: for polynonimal, G: Gauss, L: Layer

def train(sk, eps, max_update_num):
    # xi1 is the first positive value (image)
    sk.initialization(KERNEL_TYPE)
    # xj1 is the first negative value (image)
    # m is form slide 5
    # method for prime
    #
    # From this point, I assume the initialization step is completed. So all
    #c_flag = False
    adapt_count = 0
    is_stop = False
    while not is_stop and adapt_count < max_update_num:
        is_stop, t = sk.stop(eps)
        # The stop function check the model convergence < epsilon
        if not is_stop:

            x = sk.xip[t]
            sk.adapt(adapt_count,t)
            # Repeat the process until convergence < epsilon or
            # more than max_updates adaptation steps have been done
            # (adapt_count >= max_update
        else:
            lamb_da_t = sk.lamb_da
        adapt_count += 1
    if adapt_count >= max_update_num:
        print "Max updates reached."

    alpha_pair = []
    for i in sk.alpha:
       if sk.alpha[i] != 0:
            y = 0
            if i in sk.Ip:
                y = 1
            alpha_pair.append([i,float(sk.alpha[i]),y])
    print adapt_count
    result = ""
    result += str(sk.mp)+","+str(sk.mn)+","+str(sk.lamb_da)+"\n"
    result += str(sk.A)+","+str(sk.B)
    for alph in alpha_pair:
        result+=str(alph[0])+str(alph[1])+"\n"
    f = open(model_file_name,'w')
    f.write(result)
    f.close()


args = sys.argv
epsilon = args[1]
max_updates = args[2]
class_letter = args[3]
model_file_name = args[4]
train_folder_name = args[5]

sk = SKAlgo()
sk.read(class_letter, train_folder_name)
train(sk, epsilon, max_updates)
