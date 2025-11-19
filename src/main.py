import numpy as np
from utils import *
from dataplot import *    
from configure import *
import os
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument('--N', type=int, help='number of ions', default=50)
Parser.add_argument('--t0', type=float, help='the beginning of evolution', default=0.0)
Parser.add_argument('--config_name', type=str, help='the name of voltage configs', default="flat_28")
Parser.add_argument('--time', type=float, help='total simulation time in microseconds', default=np.inf)
Parser.add_argument('--epochs',  type=int, default=10, help='number of optimization epochs')
Parser.add_argument('--CUDA', action='store_true', help='use CUDA for computation')
Parser.add_argument('--plot', action='store_true', help='enable plotting')
Parser.add_argument('--interval', type=float, help='the interval between 2 adjacent frames', default=0.5)
Parser.add_argument('--save_final', type=bool, help='enable saving the final configuration', default=False)
Parser.add_argument('--save_traj', type=bool, help='enable saving the trajectory', default=False)

dirname = os.path.dirname(__file__)

flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
filename=os.path.join(dirname, "../../../data/monolithic20241118.csv") #文件名：导入的电势场格点数据
basis_filename=os.path.join(dirname, "electrode_basis.json")#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5
def oscillate(t):
    return np.cos(2*t)

if __name__ == "__main__":
    # V_static = {"RF": -7.25156238001962,
    #     "U1": 0.17269838528176118,
    #     "U2": 0.43617245456405995,
    #     "U3": 0.611378067391062,
    #     "U4": -0.886486757350588,
    #     "U5": 0.9771554372897736,
    #     "U6": 0.5212677070289424,
    #     "U7": 0.22461453785257968}
    # V_dynamic = {"RF":300}
    args = Parser.parse_args()
    device = 1 if args.CUDA else 0
    print("Using %s for computation."%( "CUDA" if device==1 else "CPU"))
    ini_range = np.random.randint(100, 200) 
    N = args.N  
    charge = np.ones(N) 
    mass = np.ones(N)
    t_start = args.t0
    interval = args.interval
    config_name = args.config_name
    basis = Data_Loader(filename, basis_filename, flag_smoothing)
    basis.loadData()
    configure = Configure(basis=basis)
    # configure.load_from_file(os.path.join(dirname, "../saves/saved_config_regression_0.01_1000.json"))
    configure.load_from_file(os.path.join(dirname, "../saves/%s.json"%config_name))  
    # configure.load_from_param(V_static, V_dynamic)
    t = args.time
    std_y, len_z, simu_t = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=interval, batch=50, t=t, device=device, plotting=args.plot, t_start=t_start, config_name=config_name, save_final=args.save_final, save_traj=args.save_traj)
    print("Estimated thickness: %.3f um at time %.3f us."%(std_y, simu_t))