import numpy as np
from utils import *
from dataplot import *    
from configure import *
import os
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument('--N', type=int, help='number of ions', default=50)
Parser.add_argument('--electrodes', type=int, help='number of electrodes', default=28)
Parser.add_argument('--t0', type=float, help='the beginning of evolution', default=0.0)
Parser.add_argument('--config_name', type=str, help='the name of voltage configs', default="flat_28")
Parser.add_argument('--time', type=float, help='total simulation time in microseconds', default=np.inf)
Parser.add_argument('--epochs',  type=int, default=10, help='number of optimization epochs')
Parser.add_argument('--CUDA', action='store_true', help='use CUDA for computation')
Parser.add_argument('--plot', action='store_true', help='enable plotting')
Parser.add_argument('--interval', type=float, help='the interval between 2 adjacent frames', default=1)
Parser.add_argument('--g', type=float, help='cooling rate', default=0.1)
Parser.add_argument('--isotope', type=str, help='isotope type', default="Ba135")
Parser.add_argument('--save_final', action='store_true', help='enable saving the final configuration')
Parser.add_argument('--save_traj', action='store_true', help='enable saving the trajectory')
Parser.add_argument('--bilayer', action='store_true', help='the trap is designed to trap 2 layers')
# Parser.add_argument('--save_final', type=bool, help='enable saving the final configuration', default=False)
# Parser.add_argument('--save_traj', type=bool, help='enable saving the trajectory', default=False)

dirname = os.path.dirname(__file__)

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
    flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
    if args.bilayer:
        #Using bilayer
        filename=os.path.join(dirname, "../../data/bilayer_both_x50y250z200_dx2dy2dz5_err1e-4_fine.csv")
        basis_filename=os.path.join(dirname, "bilayer_basis.json")
        sym = False
        print("Bilayer...")
    elif args.electrodes == 28:
        filename=os.path.join(dirname, "../../../data/monolithic20241118.csv") #文件名：导入的电势场格点数据
        # filename=os.path.join(dirname, "../../../data/28electrodes_x60y40z1000.csv") #文件名：导入的电势场格点数据
        basis_filename=os.path.join(dirname, "electrode_basis.json")#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5
        sym = False
    elif args.electrodes == 60:
        # filename=os.path.join(dirname, "../../../data/60electrodes_x50y20z1000_tiny.csv")
        filename=os.path.join(dirname, "../../../data/60electrodes_sym4_x100y50z1000_dx2dy2dz5_CM62.csv")
        basis_filename=os.path.join(dirname, "electrode_basis_60.json")
        sym = False
    device = 1 if args.CUDA else 0
    print("Using %s for computation."%( "CUDA" if device==1 else "CPU"))
    ini_range = np.random.randint(100, 200) 
    N = args.N  
    charge = np.ones(N) 
    mass = np.ones(N)
    # #掺杂同位素离子
    # mass[:100] = 133/135
    # mass[100:200] = 134/135
    # mass[200:300] = 136/135
    # mass[300:400] = 137/135
    # mass[400:500] = 138/135
    t_start = args.t0
    interval = args.interval
    config_name = args.config_name
    basis = Data_Loader(filename, basis_filename, flag_smoothing)
    basis.loadData()
    configure = Configure(basis=basis, sym=sym, g=args.g, isotope=args.isotope)
    # configure.load_from_file(os.path.join(dirname, "../saves/saved_config_regression_0.01_1000.json"))
    configure.load_from_file(os.path.join(dirname, "../saves/%s.json"%config_name))  
    # configure.load_from_param(V_static, V_dynamic)
    t = args.time
    if args.bilayer:
        stdy_upper, stdy_lower, len_z, simu_t = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=interval, batch=50, t=t, device=device, plotting=args.plot, t_start=t_start, config_name=config_name, save_final=args.save_final, save_traj=args.save_traj, bilayer=args.bilayer)
        print("Estimated thickness: Upper %.3f um, Lower %.3f um at time %.3f us."%(stdy_upper, stdy_lower, simu_t))
    else:
        std_y, len_z, simu_t = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=interval, batch=50, t=t, device=device, plotting=args.plot, t_start=t_start, config_name=config_name, save_final=args.save_final, save_traj=args.save_traj, bilayer=args.bilayer)
        print("Estimated thickness: %.3f um at time %.3f us."%(std_y, simu_t))