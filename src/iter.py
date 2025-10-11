import subprocess
from dataplot import *

N = 1500

def LossFunction(U):
    cmd = "python3 monolithic.py --N %d"%N
    for i in range(len(U)):
        cmd += " --U%s %.2f"%(str(i), U[i])
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    L = np.load('../data_cache/2D/loss_%d.npy'%N)[-1][-1]
    return L

def grad(L, U, round):
    dL = np.zeros_like(U)
    h = 0.1
    for i in range(U.shape[0]):
        cmd = "python3 monolithic.py --N %d"%N
        for j in range(U.shape[0]):
            if i == j:
                cmd += " --U%s %.2f"%(str(i), U[i]+h)
            else:
                cmd += " --U%s %.2f"%(str(j), U[j])
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        new_L = np.load('../data_cache/2D/loss_%d.npy'%N)[-1][-1]
        # np.save('../data_cache/iter%d/%d/loss_%d.npy'%(N, i, round), new_L)
        dL[i] = new_L - L
    return dL/h

if __name__ == "__main__":
    rounds = 40
    U = [-6.8, 0, 0, 0, -3.0]
    # U = [-6.4, 0, 0, 0, -0.5]
    # U = [-6.89769083,  0.04370838,  0.01557981,  0.11838954, -0.1673234]
    # # L = 0.7186347613127857
    # UL = np.load('../data_cache/iter%d/UL30.npy'%(N))[-1]
    # U = UL[:-1]
    L = LossFunction(U)
    print(f'U = {U}, L = {L}')
    start = 0
    # U, L = np.load('../data_cache/iter%d/UL.npy'%(N))[-1][:-1], np.load('../data_cache/iter%d/UL.npy'%(N))[-1][-1]
    UL = np.zeros((rounds-start, len(U)+1))
    for round in range(start, rounds):
        grad_L = grad(L, np.array(U), round)
        # lr = 0.01 #0.1/(round+1)**0.5
        # lr = 0.01*(1-(round-start)/(rounds-start))**0.5 + 0.002
        lr = 0.1*np.exp(-round/(rounds-start)) # Exponential decay learning rate
        U = U - lr*grad_L
        L = LossFunction(U)
        # np.save('../data_cache/iter%d/UL_%d.npy'%(N, round), np.append(U, L)) 
        UL[round-start] = np.append(U, L) # A 1*6 vector, with the first 5 elements being U0~U4, and the last element being L
        print(f'round {round}, U = {U}, L = {L}, grad_L = {grad_L}')
    np.save('../data_cache/iter%d/UL%d-%d_lrexp_newinit.npy'%(N, start, rounds), UL)