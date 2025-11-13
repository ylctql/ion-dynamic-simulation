import multiprocessing as mp
import numpy as np
import pandas as pd   
import ionsim
from utils import *
from dataplot import *    
import math,csv,json
from scipy.signal import savgol_filter
from scipy.constants import e, pi, epsilon_0
import types
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import psutil
import time

freq_RF = 35.28 #RF射频频率@MHz
Omega = freq_RF*2*pi*10**6 #RF射频角频率@SI
epsl = epsilon_0#8.854*10**(-12)#真空介电常数@SI
m = 2.273e-25 #Ba137+ #170.936*(1.66053904*10**(-27))#离子的质量 @SI #Yb171=170.936323；Yb174=173.938859
ec = e#1.6*10**(-19)#元电荷@SI
dt = 2/Omega#单位时间dt #dt*T*Omega=2pi ->  T=pi
dl = (( ec**2)/(4*pi*m*epsl*(Omega)**2))**(1/3)#单位长度dl
dV = m/ec*(dl/dt)**2 #单位电压dV

class Data_Loader:
    '''
    Load electrode potential data from CSV file and custom basis configuration from JSON file.

    :param filename: path to the CSV file containing electrode potential data
    :param basis_filename: path to the JSON file containing custom basis configuration
    :param smoothing: whether to apply smoothing to the potential data
    '''

    def __init__(self, filename: str, basis_filename: str, smoothing: bool) -> None:
        self.filename = filename
        self.basis_filename = basis_filename
        self.flag_smoothing = smoothing
        self.unit_l = 1e-3
        self.units = {"m":1e0,"cm":1e-2,"mm":1e-3,"um":1e-6}
        self.data: np.ndarray
        self.keymaps: dict
        self.basisGroup_map: dict
        self.dl = dl
        self.dV = dV

    def smoothing(self, data):#由于电势场数据崎岖不平，因而做的平滑化函数
        return savgol_filter(data, 11, 3)
    
    def getcol(self,key): #把电势场名key转换为表格的列序号
        if key in self.keymaps:
            return self.keymaps[key]
        else:            
            cols=[v for k,v in self.keymaps.items() if (key == k.split(".")[0])]
            return cols[0] if len(cols)>0 else None
        
    def loadData(self,name=None):#读取加载电势场格点文件数据
        try:
            self.load_Settings_CustomBasis()
            print("加载自定义Basis设置")
        except Exception as er:
            print(er)
        if name is None:
            name = self.filename
        with open(name,encoding='utf-8',mode='r+') as file_read:
            csvread = csv.reader(file_read)
            for i,row in enumerate(csvread):
                if i > 20:
                    break
                if row[0] == r'% Length unit':
                     self.unit_l = self.units[row[1]]
                     print("self.unit_l=", self.unit_l)
                if row[0] == r'% x':
                    self.keymaps = {row[v].replace(r"% ", ""): v for v in range(len(row))}
                    self.keynames = [name for name in self.keymaps if name not in ["x", "y", "z"]]
                    break
        dat = pd.read_csv(name, comment='%', header=None)
        data = dat.to_numpy()
        data[:, 3:] *= 1 / self.dV  # csv文件内的电势单位：V
        data[:, 0:3] *= self.unit_l / self.dl  # csv文件内坐标的长度单位：mm

        data = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]  # index统一排序使得格点数据与文件内坐标顺序无关
        self.coordinate = [self.x, self.y, self.z] = [x, y, z] = [np.unique(data[:, i]) for i in range(3)]
        self.coordinate_um = [temp / (1e-6 / self.dl) for temp in self.coordinate]
        self.data = data

    def load_basis(self, key):  # 加载电势场名key的格点数据
        coln = self.getcol(key)

        if coln is None:
            components = self.basisGroup_map[key]
            outputs = self.load_basis(components)
        else:
            outputs = self.data[:, coln].reshape(len(self.x), len(self.y), len(self.z))
        
        if self.flag_smoothing and self.smoothing is not None:
            outputs = self.smoothing(outputs)
        return outputs       

    def load_Settings_CustomBasis(self):#加载自定义Basis设置
        self.basisGroup_map = self.loadConfig() 
        print("Loaded custom basis:", self.basisGroup_map)

    def loadConfig(self):
        with open(self.basis_filename, 'r',encoding='utf-8') as jf:
            config = json.load(jf)
        return config

class Configure:
    """
    The configuration of static and dynamic voltages.

    :param V_static: dictionary of static voltages, key is electrode name, value is voltage in volts
    :param V_dynamic: dictionary of dynamic voltages, key is electrode name, value is a list where the first element is amplitude in volts and the second element is a function of time
    :param basis: Data_Loader object containing electrode potential data and basis configuration
    """

    def __init__(self, basis: Data_Loader, V_static: dict = {}, V_dynamic: dict = {}) -> None:

        self.V_static = V_static
        self.V_dynamic = V_dynamic
        self.key_list = list(self.V_static.keys())
        self.num_electordes = len(self.key_list)
        self.basis = basis
        self.grids_dc: list = []
        self.grids_rf: dict = {}
        self.grad_dc: dict = {}
        self.grad_rf: dict = {}
        self.dV = dV
        self.ec = ec
        self.Omega = Omega
        self.m = m
        self.dt = dt
        self.dl = dl

    def __reduce__(self):
        return (
            self.__class__, 
            (self.basis, self.V_static, self.V_dynamic), 
            {"grids_dc": self.grids_dc, "grids_rf": self.grids_rf}
        )

    def __setstate__(self, state):
        self.grids_dc = state.get("grids_dc", [])
        self.grids_rf = state.get("grids_rf", {})

        if not self.grids_dc or not self.grids_rf:
            self.calc_field()

    def load_from_file(self, filename: str) -> None:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.V_static = data.get("V_static", {})
            self.V_dynamic = data.get("V_dynamic", {})
    
    def load_from_param(self, V_static: dict, V_dynamic: dict) -> None:
        self.V_static = V_static
        self.V_dynamic = V_dynamic

    def _interpret_voltage(self, value):
        if type(value) == list:
            return value[0]
        return value

    def _interpret_dynamic(self, t):
        return np.cos(2*t)

    def _gen_grids(self, potential_static):
        [x, y, z] = self.basis.coordinate
        fieldx, fieldy, fieldz = np.gradient(-potential_static, x, y, z, edge_order=2)
        grid_x = ionsim.Grid(x, y, z, value=fieldx)
        grid_y = ionsim.Grid(x, y, z, value=fieldy)
        grid_z = ionsim.Grid(x, y, z, value=fieldz)
        return [grid_x, grid_y, grid_z]
    
    def pseudo_potential(self):
        V0 = self._interpret_voltage(self.V_dynamic["RF"])*self.basis.load_basis("RF")*self.ec*self.dV #此处统一使用国际单位制
        [x, y, z] = self.basis.coordinate_um   #换算成国际单位制
        Fx, Fy, Fz = np.gradient(-V0, x, y, z, edge_order=2)   
        # F0 = np.sqrt(Fx**2 + Fy**2 + Fz**2)*1e6 #因为距离单位为um，所以梯度要乘1e6
        V_pseudo_rf = (Fx**2 + Fy**2 + Fz**2)*1e12/(4*self.m*self.Omega**2*self.ec)
        return V_pseudo_rf
    
    def static_potential(self):
        potential_static = 0
        for key, value in self.V_static.items():
            potential_static += self.basis.load_basis(key) * self._interpret_voltage(value)
        return potential_static*self.dV
    
    def total_potential(self):
        return self.static_potential() + self.pseudo_potential()

    def calc_field(self) -> None:
        potential_static = 0
        for key, value in self.V_static.items():
            potential_static += self.basis.load_basis(key) * self._interpret_voltage(value)
        self.grids_dc = self._gen_grids(potential_static)

        potential_dynamic = []
        grids_dynamic_dict = {}
        for key, value in self.V_dynamic.items():
            potential_dynamic.append(self.basis.load_basis(key) * self._interpret_voltage(value))
            grids_dynamic_dict[key] = [self._gen_grids(potential_dynamic[-1]), value]
        self.grids_rf = grids_dynamic_dict

    def calc_force(self, r: np.ndarray, v: np.ndarray, t: float):
        gamma = 0.1
        bound_min = [np.min(self.basis.coordinate[i]) + 1e-9 for i in range(3)]
        bound_max = [np.max(self.basis.coordinate[i]) - 1e-9 for i in range(3)]

        f = np.zeros_like(r)
        mask = self.grids_dc[0].in_bounds(r)#大家都没出界吧？
        if len(np.where(mask == False)[0]) > 0:
            # print("警告出界")    
            r = np.array([np.clip(r[:, i], bound_min[i], bound_max[i]) for i in range(3)]).T#如果出界，就按边界上的电场
            mask =  self.grids_dc[0].in_bounds(r)
        r_mask = r[mask].copy(order='F')
        
        # inside bounds
        coord = self.grids_dc[0].get_coord(r_mask)
        f_in = (np.vstack(tuple([grid.interpolate(coord) for grid in self.grids_dc])))#静电力
        for key, value in self.grids_rf.items():
            grids_rf = value[0]
            f_in = f_in + np.vstack(tuple([grid.interpolate(coord) for grid in grids_rf])) * self._interpret_dynamic(t)#加上含时的力
        f_in = f_in.transpose()
        f[mask] = f_in
        # outside bounds# r_nmask = r[~mask].copy(order='F')# f[~mask] = np.zeros_like(r_nmask)#一般来说这里为空
        f = f - gamma * v
        return f

    def simulation(self, N: int, ini_range: int, mass: np.ndarray, charge: np.ndarray, step: int, interval: int, batch: int, t: float, device: bool, plotting: bool, alpha: float = 1.0) -> tuple:

        backend = CalculationBackend(device=device, step=step, interval=interval, batch=batch, time=t/(self.dt * 1e6))

        q1 = mp.Queue()
        q2 = mp.Queue(maxsize=50)

        r0 = (np.random.rand(N, 3)-0.5) *ini_range
        v0 = np.zeros((N, 3))

        q1.put(Message(CommandType.START, r0, v0, mass, charge, self.calc_force))
        q2.put(Frame(r0, v0, 0))

        proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)   
        proc.start()

        f = None
        if plotting:
            plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.04, z_range=550, x_range=100, y_range=20, z_bias=0, dl=self.dl*1e6,dt=self.dt*1e6)
            f = plot.start()
            if not plot.is_alive():
                q1.put(Message(CommandType.STOP))
                while True:
                    new_f = q2.get()
                    if isinstance(new_f, bool) and not new_f:
                        break
                proc.join()
            std_y = f.r[:, 1].std() * self.dl * 1e6
        else:
            count = 0
            std_y = 0.0
            dominator = 0.0
            while True:
                new_f = q2.get()
                if isinstance(new_f, bool) and not new_f:
                    break
                f = new_f
                if count>2000:  # From experience, about 200 data points
                    std_y += f.r[:, 1].std() * self.dl * 1e6
                    dominator += 1
                count += 1
            proc.join()
            # print("max count:", count)
            std_y /= dominator
            # print(std_y)
        len_z = np.abs(f.r[:, 2].max() - f.r[:, 2].min()) * self.dl * 1e6
        return std_y, len_z, f.timestamp * self.dt * 1e6
    
    def single_grad(self, key_id, N: int, ini_range: int, mass: np.ndarray, charge: np.ndarray, step: int, interval: int, batch: int, t: float, device: bool, h_dc: float = 0.01, h_rf: float = 0.1, r: float = 0.05, sym: bool = True, biside: bool = False) -> None:
        key = self.key_list[key_id]
        original_value = self.V_static[key]
        self.V_static[key] = original_value + h_dc
        if key!="RF" and sym:
            self.V_static[self.key_list[self.num_electordes-key_id]] = self.V_static[key]
        self.calc_field()
        std_plus, len_plus, _ = self.simulation(N, ini_range, mass, charge, step, interval, batch, t, device, plotting=False)
        if biside:
            self.V_static[key] = original_value - h_dc
            if key!="RF" and sym:
                self.V_static[self.key_list[self.num_electordes-key_id]] = self.V_static[key]
            self.calc_field()
            std_minus, len_minus, _ = self.simulation(N, ini_range, mass, charge, step, interval, batch, t, device, plotting=False)
            grad = (std_plus - std_minus + r*len_plus - r*len_minus) / (2 * h_dc)
        else:
            grad = (std_plus + r*len_plus)/h_dc
        self.V_static[key] = original_value
        return grad

    def calc_gradient(self, N: int, ini_range: int, mass: np.ndarray, charge: np.ndarray, step: int, interval: int, batch: int, t: float, device: bool, h_dc: float = 0.01, h_rf: float = 0.1, r: float = 0.05, sym: bool = True, biside: bool = False, cpu_parellel: bool = False) -> None:
        start = time.time()
        if cpu_parellel:
            with ProcessPoolExecutor() as executor:
                Futures = [executor.submit(self.single_grad, key_id, N, ini_range, mass, charge, step, interval, batch, t, device, h_dc, h_rf, r, sym, biside) for key_id in range(int(self.num_electordes/2+1))]
                max_workers = executor._max_workers
                print(f"最大并行进程数: {max_workers}")
                parent = psutil.Process()  # 当前 Python 进程
                children = parent.children(recursive=True)  # 获取所有子进程
                print(f"当前运行的子进程数 by psutil: {len(children)}")
                time.sleep(0.1)  # 短暂等待以确保子进程启动
                active_processes = len(multiprocessing.active_children())
                print(f"当前活跃进程数 by multiprocessing: {active_processes}")
                for key_id in range(int(self.num_electordes/2+1)):
                    self.grad_dc[self.key_list[key_id]] = Futures[key_id].result()
            if sym:
                for id in range(int(self.num_electordes/2+1), self.num_electordes):
                    self.V_static[self.key_list[id]] = self.V_static[self.key_list[self.num_electordes-id]]
                    self.grad_dc[self.key_list[id]] = self.grad_dc[self.key_list[self.num_electordes-id]]
                    print("%s = %s"%(self.key_list[id], self.key_list[self.num_electordes-id]))
        else:
            for key_id, key in enumerate(self.V_static.keys()):
                if sym and key_id > self.num_electordes/2:
                    for id in range(key_id, self.num_electordes):
                        self.V_static[self.key_list[id]] = self.V_static[self.key_list[self.num_electordes-id]]
                        self.grad_dc[self.key_list[id]] = self.grad_dc[self.key_list[self.num_electordes-id]]
                        print("%s = %s"%(self.key_list[id], self.key_list[self.num_electordes-id]))
                    break
                self.grad_dc[key] = self.single_grad(key_id, N, ini_range, mass, charge, step, interval, batch, t, device, h_dc, h_rf, r, sym, biside)
                # self.V_static[key] = original_value
                print("finished %s..."%key)
        end = time.time()
        print("finished all static keys...")
        print("CPU_parellel: %s"%cpu_parellel, "using %.3f s"%(end-start))
        for key in self.V_dynamic.keys():
            original_value = self.V_dynamic[key]
            self.V_dynamic[key] = original_value + h_rf
            self.calc_field()
            std_plus, len_plus, _ = self.simulation(N, ini_range, mass, charge, step, interval, batch, t, device, plotting=False)
            if biside:
                self.V_dynamic[key] = original_value - h_rf
                self.calc_field()
                std_minus, len_minus, _ = self.simulation(N, ini_range, mass, charge, step, interval, batch, t, device, plotting=False)
                grad = (std_plus - std_minus + r*len_plus - r*len_minus) / (2 * h_rf)
            else:   
                grad = (std_plus + r*len_plus)/h_rf
            self.grad_rf[key] = grad
            self.V_dynamic[key] = original_value
        print("finished all dynamic keys...")

    def update(self, lr_dc: float = 0.01, lr_rf: float = 0.5, gamma:float =0.99, epoch:int = 0):
        for key in self.grad_dc.keys():
            self.V_static[key] -= lr_dc * self.grad_dc[key] * gamma**epoch
        for key in self.grad_rf.keys():
            self.V_dynamic[key] -= lr_rf * self.grad_rf[key]

    def save(self, filename: str) -> None:
        V_static = {}
        V_dynamic = {}
        for key, value in self.V_static.items():
            V_static[key] = float(value)
        for key, value in self.V_dynamic.items():
            V_dynamic[key] = float(value)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "V_static": V_static,
                "V_dynamic": V_dynamic
            }, f, indent=4)

