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
filename=os.path.join(dirname, "../../autodl-tmp/monolithic20241118.csv") #文件名：导入的电势场格点数据
basis_filename=os.path.join(dirname, "electrode_basis.json")#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5

dt = Const.dt
dl = Const.dl

if __name__ == "__main__":
    args = Parser.parse_args()
    device = 1 if args.CUDA else 0
    print("Using %s for computation."%( "CUDA" if device==1 else "CPU"))
    ini_range = np.random.randint(100, 200) 
    N = args.N  
    charge = np.ones(N) 
    mass = np.ones(N) 
    basis = Data_Loader(filename, basis_filename, flag_smoothing)
    basis.loadData()
    configure = Configure(basis=basis, V_dynamic={"RF": 275}, V_static={"RF": -6.4,
        "U1": 0.0,
        "U2": 0.0,
        "U3": 0.0,
        "U4": -0.5,
        "U5": 0.0,
        "U6": 0.0,
        "U7": 0.0})
    # configure.load_from_file(os.path.join(dirname, "../saves/test_config.json"))
    t = args.time
    r = np.load(os.path.join(dirname, "../saves/final_positions_300.npy")) / dl / 1e3
    f = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=5, batch=50, t=t, device=device, plotting=args.plot, r_init = r)
    std_y = f.r[:, 1].std() * dl * 1e6
    len_z = np.abs(f.r[:, 2].max() - f.r[:, 2].min()) * dl * 1e6
    f.save_position(os.path.join(dirname, "../saves/positions_300.npy"))
    print("Estimated thickness: %.3f um at time %.3f us."%(std_y, f.timestamp * dt * 1e6))