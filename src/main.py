import numpy as np
from utils import *
from dataplot import *    
from configure import *
import os
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument('--N', type=int, help='number of ions', default=50)
Parser.add_argument('--time', type=float, help='total simulation time in microseconds', default=np.inf)
Parser.add_argument('--epochs',  type=int, default=10, help='number of optimization epochs')
Parser.add_argument('--CUDA', action='store_true', help='use CUDA for computation')
Parser.add_argument('--plot', action='store_true', help='enable plotting')

dirname = os.path.dirname(__file__)

flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
filename=os.path.join(dirname, "../data/monolithic20241118.csv") #文件名：导入的电势场格点数据
basis_filename=os.path.join(dirname, "electrode_basis.json")#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5
def oscillate(t):
    return np.cos(2*t)

if __name__ == "__main__":
    args = Parser.parse_args()
    device = 1 if args.CUDA else 0
    print("Using %s for computation."%( "CUDA" if device==1 else "CPU"))
    ini_range = np.random.randint(100, 200) #初始范围也随机，探索更多可能
    N = args.N  #离子数
    charge = np.ones(N) #每个离子带电荷量都是1个元电荷
    mass = np.ones(N) #每个离子质量都是1m，具体大小见下面的m
    basis = Data_Loader(filename, basis_filename, flag_smoothing)
    basis.loadData()
    configure = Configure(V_static={
        "RF": -6.961206877539749,
        "U1": 0.04790252101741849,
        "U2": -0.07065312492226086,
        "U3": 0.14516592808150125,
        "U4": 1.4597137709304562,
        "U5": 0.2083854172475653,
        "U6": -0.04400262941551117,
        "U7": 0.04874354976365938
    }, V_dynamic={"RF": [274.9213508142432, oscillate]}, basis=basis)#静态电压和动态电压
    t = args.time
    std, simu_t = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=5, batch=50, t=t, device=device, plotting=args.plot)
    print("Estimated thickness: %.3f um at time %.3f us."%(std, simu_t))