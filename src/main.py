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
Parser.add_argument('--config_name', type=str, help='the name of voltage configs', default="cpu_28")
Parser.add_argument('--time', type=float, help='total simulation time in microseconds', default=np.inf)
Parser.add_argument('--plot', action='store_true', help='enable plotting')
Parser.add_argument('--interval', type=float, help='the interval between 2 adjacent frames', default=1)
Parser.add_argument('--g', type=float, help='cooling rate', default=0.1)
Parser.add_argument('--isotope_type', type=str, help='isotope type', default="Ba135")
Parser.add_argument('--alpha', type=float, help='doping ratio', default=0.1)
Parser.add_argument('--save_final', action='store_true', help='enable saving the final configuration')
Parser.add_argument('--save_traj', action='store_true', help='enable saving the trajectory')
Parser.add_argument('--bilayer', action='store_true', help='the trap is designed to trap 2 layers')
Parser.add_argument('--save_final_image', type=str, help='path to save the final frame image', default=None)

dirname = os.path.dirname(__file__)

def oscillate(t):
    return np.cos(2*t)

if __name__ == "__main__":
    args = Parser.parse_args()
    flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
    if args.bilayer:
        #Using bilayer
        filename=os.path.join(dirname, "../../../data/bilayer_both_x50y250z200_dx2dy5dz5.csv")
        basis_filename=os.path.join(dirname, "bilayer_basis.json")
        sym = False
        print("Bilayer...")
    elif args.electrodes == 28:
        filename=os.path.join(dirname, "../../data/monolithic20241118.csv") #文件名：导入的电势场格点数据
        basis_filename=os.path.join(dirname, "electrode_basis.json")#文件名：自定义Basis设置
        sym = False
    elif args.electrodes == 60:
        filename=os.path.join(dirname, "../../../data/60electrodes_sym4_x100y50z1000_dx2dy2dz5_CM62.csv")
        basis_filename=os.path.join(dirname, "electrode_basis_60.json")
        sym = False
    device = 0  # CPU only for ism-cpu
    print("Using CPU for computation.")
    ini_range = np.random.randint(100, 200) 
    N = args.N  
    charge = np.ones(N) 
    mass = np.ones(N)
    # 参杂离子比例为alpha
    # 分配方案：Ba133, Ba134, Ba136, Ba137, Ba138各占alpha比例，Ba135占据剩余位置（1-5*alpha比例）
    alpha = args.alpha
    if alpha > 0:
        mass[:int(N*alpha)] = 10/135  # Ba133
        mass[int(N*alpha):int(2*N*alpha)] = 134/135  # Ba134
        mass[int(2*N*alpha):int(N*(1-3*alpha))] = 1.0  # Ba135（占据剩余位置）
        mass[int(N*(1-3*alpha)):int(N*(1-3*alpha)+N*alpha)] = 136/135  # Ba136
        mass[int(N*(1-3*alpha)+N*alpha):int(N*(1-3*alpha)+2*N*alpha)] = 137/135  # Ba137
        mass[int(N*(1-3*alpha)+2*N*alpha):N] = 1000/135  # Ba138
    t_start = args.t0
    interval = args.interval
    config_name = args.config_name
    basis = Data_Loader(filename, basis_filename, flag_smoothing)
    basis.loadData()
    configure = Configure(basis=basis, sym=sym, g=args.g, isotope_type=args.isotope_type)
    # 尝试从saves目录加载配置文件
    config_file = os.path.join(dirname, "../saves/%s.json"%config_name)
    if os.path.exists(config_file):
        configure.load_from_file(config_file)
    else:
        # 如果saves目录不存在，尝试从当前目录加载
        config_file = os.path.join(dirname, "%s.json"%config_name)
        if os.path.exists(config_file):
            configure.load_from_file(config_file)
        else:
            print(f"警告: 配置文件 {config_file} 不存在，使用默认配置")
    configure.calc_field()
    t = args.time
    # if args.bilayer:
    #     stdy_upper, stdy_lower, len_z, simu_t = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=interval, batch=50, t=t, device=device, plotting=args.plot, t_start=t_start, config_name=config_name, save_final=args.save_final, save_traj=args.save_traj, bilayer=args.bilayer, save_image_path=args.save_final_image)
    #     print("Estimated thickness: Upper %.3f um, Lower %.3f um at time %.3f us."%(stdy_upper, stdy_lower, simu_t))
    # else:
    #     std_y, len_z, simu_t = configure.simulation(N=N, ini_range=ini_range, mass=mass, charge=charge, step=10, interval=interval, batch=50, t=t, device=device, plotting=args.plot, t_start=t_start, config_name=config_name, save_final=args.save_final, save_traj=args.save_traj, bilayer=args.bilayer, save_image_path=args.save_final_image)
    #     print("Estimated thickness: %.3f um at time %.3f us."%(std_y, simu_t))
    grids_dc = configure.grids_dc
    grids_rf = configure.grids_rf
    basis = configure.basis
    
    def force(r: np.ndarray, v: np.ndarray, t: float):
        
        gamma = 0.1
        bound_min = [np.min(basis.coordinate[i]) + 1e-9 for i in range(3)]
        bound_max = [np.max(basis.coordinate[i]) - 1e-9 for i in range(3)]
        mask = grids_dc[0].in_bounds(r)#大家都没出界吧？
        if len(np.where(mask == False)[0]) > 0:
        # print("警告出界")    
            r = np.array([np.clip(r[:, i], bound_min[i], bound_max[i]) for i in range(3)]).T#如果出界，就按边界上的电场
            mask =  grids_dc[0].in_bounds(r)
        r_mask = r[mask].copy(order='F')
        f = np.zeros_like(r)
        # inside bounds
        coord = grids_dc[0].get_coord(r_mask)
        f_in = (np.vstack(tuple([grid.interpolate(coord) for grid in grids_dc])))#静电力
        for key, value in grids_rf.items():
            current_grids_rf = value[0]
            f_in = f_in + np.vstack(tuple([grid.interpolate(coord) for grid in current_grids_rf])) * np.cos(2*t)#加上含时的力
        f_in = f_in.transpose()
        f[mask] = f_in
        # outside bounds# r_nmask = r[~mask].copy(order='F')# f[~mask] = np.zeros_like(r_nmask)#一般来说这里为空
        # k_unit = np.array([0, np.cos(0.25*np.pi), np.sin(0.25*np.pi)])
        # f = f - gamma * np.dot(np.dot(v, k_unit).reshape(-1, 1), k_unit.reshape(1, -1))
        f = f - gamma * v
        return f

    ini_range = np.random.randint(100, 200) #初始范围也随机，探索更多可能
    r0 = (np.random.rand(N, 3)-0.5) *ini_range
    v0 = np.zeros((N, 3))
    r0[:, 1] *= 0.1
    print("using y0.1 initial condition")
    
    step=10
    batch=50
    dt = configure.dt
    dl = configure.dl
    dV = configure.dV
    backend = CalculationBackend(device=device, step=step, interval=interval, batch=batch, time=t/(dt * 1e6), dt=dt, dl=dl, dV=dV, config_name=config_name, isotope_type=args.isotope_type, g=args.g)

    q1 = mp.Queue()
    q2 = mp.Queue(maxsize=50)
    # 设置同位素 - 优先使用--alpha参数，如果没有则使用--isotope_ratio
    isotope_ratio = args.alpha if args.alpha is not None else args.isotope_ratio  # 五种同位素的比例
    if isotope_ratio > 0:
        mass[:int(N*isotope_ratio)] = 10/135  # Ba133
        mass[int(N*isotope_ratio):int(2*N*isotope_ratio)] = 134/135  # Ba134
        mass[int(2*N*isotope_ratio):int(N*(1-3*isotope_ratio))] = 1.0  # Ba135（占据剩余位置）
        mass[int(N*(1-3*isotope_ratio)):int(N*(1-3*isotope_ratio)+N*isotope_ratio)] = 136/135  # Ba136
        mass[int(N*(1-3*isotope_ratio)+N*isotope_ratio):int(N*(1-3*isotope_ratio)+2*N*isotope_ratio)] = 137/135  # Ba137
        mass[int(N*(1-3*isotope_ratio)+2*N*isotope_ratio):N] = 1000/135  # Ba138
    
    # 计算目标演化时间（单位：dt）
    target_time_dt = None
    if args.time is not None:
        target_time_dt = args.t0/(dt*1e6) + args.time/(dt*1e6)  # 转换为dt单位的时间戳
    
    q1.put(Message(CommandType.START, r0, v0, args.t0/(dt*1e6), charge, mass, force))
    q2.put(Frame(r0, v0, args.t0/(dt*1e6)))

    proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)
    proc.start()
    
    if args.plot:
        # 如果指定了--plot，则实时显示画图
        plot = DataPlotter(q2, q1, Frame(r0, v0, args.t0/(dt*1e6)), interval=0.04, x_range=100, y_range=20, z_range=200, z_bias=0, dl=dl*1e6,dt=dt*1e6, target_time_dt=target_time_dt, mass=mass)
        plot.start()#True表示只画图 #False表示只输出轨迹到/traj/文件夹# "both"表示二者都做

        if proc.is_alive():
            q1.put(Message(CommandType.STOP))
            while True:
                f = q2.get()
                if isinstance(f, bool) and not f:
                    break
                f_last = f
            proc.join()
    else:
        # 如果不画图，则只从队列中读取数据直到达到目标时间
        print("画图已禁用，仅进行计算...")
        f_last = None
        while True:
            if proc.is_alive():
                try:
                    f = q2.get(timeout=0.1)
                    if isinstance(f, bool) and not f:
                        break
                    f_last = f
                    # 检查是否达到目标时间
                    if target_time_dt is not None and f.timestamp >= target_time_dt:
                        print(f"已达到目标时间 {args.time}us，停止计算")
                        q1.put(Message(CommandType.STOP))
                        # 继续读取直到收到停止确认
                        while True:
                            f = q2.get()
                            if isinstance(f, bool) and not f:
                                break
                            f_last = f
                        break
                except:
                    continue
            else:
                # 进程已结束，读取剩余数据
                while not q2.empty():
                    f = q2.get()
                    if isinstance(f, bool) and not f:
                        break
                    f_last = f
                break
