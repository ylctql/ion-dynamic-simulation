import numpy as np
from utils import *
from dataplot import *    
from configure import *
import os
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument('--N', type=int, help='number of ions', default=50)
Parser.add_argument('--time', type=float, default=10.0, help='total simulation time in microseconds')
Parser.add_argument('--epochs',  type=int, default=10, help='number of optimization epochs')
Parser.add_argument('--CUDA', action='store_true', help='use CUDA for computation')

dirname = os.path.dirname(__file__)

flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
filename=os.path.join(dirname, "../../autodl-tmp/monolithic20241118.csv") #文件名：导入的电势场格点数据
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
    configure = Configure(basis=basis)#静态电压和动态电压
    configure.load_from_file(os.path.join(dirname, "../saves/saved_config_regression_0.01_1000.json"))
    t = args.time
    with open(os.path.join(dirname, "../logs/saved_config_0.1equicell_5000.txt"), 'w', encoding='utf-8') as log_file:
        for i in range(args.epochs):
            print("Processing epoch %d/%d"%(i+1, args.epochs))
            configure.calc_gradient(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=5, batch=50, t=t, device=device, h=0.1, r=0.01)
            configure.update(lr=0.1)
            f = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=5, batch=50, t=t, device=device, plotting=False)
            print("Epoch %d/%d done"%(i+1, args.epochs))
            print("Estimated thickness: %.3f um; length: %.3f um; equi ratio: %.3f at epoch %d"%(f.std_y(), f.len_z(), f.equi_cell(), i+1), file=log_file)
    configure.save(os.path.join(dirname, "../saves/saved_config_0.1equicell_5000.json"))
    print("Optimization completed. Final voltages:")
    print("Static Voltages:", configure.V_static)
    print("Dynamic Voltages:", {k: v for k, v in configure.V_dynamic.items()})