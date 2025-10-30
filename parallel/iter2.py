import subprocess
from dataplot import *
import argparse
import os
from compute_loss import *
import time

parser = argparse.ArgumentParser(description='Optimization iteration for electrode voltages')
parser.add_argument('--N', type=int, default=1500, help='Number of ions')
parser.add_argument('--rounds', type=int, default=40, help='Number of optimization rounds')
parser.add_argument('--U', nargs='+', type=float, default=[-6.4]+8*[0.0], help='Initial voltages on electrodes')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for gradient descent')
args = parser.parse_args()

N = args.N
rounds = args.rounds
U = args.U
lr_ini = args.lr

dir_prefix = '../data_cache/60electrodes_tiny/'

def LossFunction(U):
    loss_computer = compute_loss(U, h=0)
    L = loss_computer.loss()
    return L

def grad(L, U, round):
    h = 0.05
    loss_computer = compute_loss(U, h)
    loss = loss_computer.loss()
    dL = loss - L
    return dL/h

if __name__ == "__main__":
    L = LossFunction(U)
    print(f'U = {U}, L = {L}')
    start = 0
    mr, vr = 0, 0   # Initialize Adam optimizer variables to zero
    # U, L = np.load('../data_cache/iter%d/UL.npy'%(N))[-1][:-1], np.load('../data_cache/iter%d/UL.npy'%(N))[-1][-1]
    UL = np.zeros((rounds-start, len(U)+1))
    global_start = time.time()
    for round in range(start, rounds):
        round_start = time.time()
        grad_L = grad(L, np.array(U), round)
        # Use Adam Method to calculate gradient
        beta1, beta2 = 0.9, 0.999
        mr = beta1*mr + (1-beta1)*grad_L
        vr = beta2*vr + (1-beta2)*(grad_L**2)
        mhat = mr/(1-beta1**(round-start+1))
        vhat = vr/(1-beta2**(round-start+1))
        # lr = lr_ini*np.sqrt(1-beta2**(round-start+1))/(1-beta1**(round-start+1))
        lr = lr_ini
        U = U - lr*mhat/(np.sqrt(vhat)+1e-8)
        L = LossFunction(U)
        UL[round-start] = np.append(U, L) # A 1*6 vector, with the first 5 elements being U0~U4, and the last element being L
        print(f'round {round}, U = {U}, L = {L}, grad_L = {grad_L}')
        round_end = time.time()
        print("%d Round time:"%round, round_end - round_start)
    global_end = time.time()
    print("Total time:",global_end-global_start)
    dir_name = dir_prefix+'iter%d/'%N
    if os.path.exists(dir_name) == False:
        os.makedirs(dir_name)
    np.save(dir_name+'UL_adam_%d-%d.npy'%(start, rounds), UL)