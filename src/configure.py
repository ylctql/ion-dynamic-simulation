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
import os
import time

freq_RF = 35.28 #RF射频频率@MHz
Omega = freq_RF*2*pi*10**6 #RF射频角频率@SI
epsl = epsilon_0#8.854*10**(-12)#真空介电常数@SI
m = 2.239367e-25 # Ba135 #170.936*(1.66053904*10**(-27))#离子的质量 @SI #Yb171=170.936323；Yb174=173.938859
ec = e#1.6*10**(-19)#元电荷@SI
dt = 2/Omega#单位时间dt #dt*T*Omega=2pi ->  T=pi    # 满足Nyquist采样定理，确保频率分析的最大值刚好为Omega
dl = ((ec**2)/(4*pi*m*epsl*(Omega)**2))**(1/3)#单位长度dl
dV = m/ec*(dl/dt)**2 #单位电压dV
print("dl:", dl, "m")
print("dt:", dt, "s")
print("dV:", dV, "V")

class Data_Loader:
    '''
    Load electrode potential data from CSV file and custom basis configuration from JSON file.

    :param filename: path to the CSV file containing electrode potential data
    :param basis_filename: path to the JSON file containing custom basis configuration
    :param smoothing: whether to apply smoothing to the potential data
    '''

    def __init__(self, filename, basis_filename: str, smoothing: bool) -> None:
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
        return savgol_filter(data, 15, 3)
    
    def getcol(self,key): #把电势场名key转换为表格的列序号
        if key in self.keymaps:
            return self.keymaps[key]
        else:            
            cols=[v for k,v in self.keymaps.items() if (key == k.split(".")[0])]
            return cols[0] if len(cols)>0 else None
    
    def getData(self, file):
        with open(file,encoding='utf-8',mode='r') as file_read:     
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
        dat = pd.read_csv(file, comment='%', header=None)
        data = dat.to_numpy()
        data[:, 3:] *= 1 / self.dV  # csv文件内的电势单位：V
        data[:, 0:3] *= self.unit_l / self.dl  # csv文件内坐标的长度单位：mm

        data = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]  # index统一排序使得格点数据与文件内坐标顺序无关
        coordinate = [np.unique(data[:, i]) for i in range(3)]
        coordinate_um = [temp / (1e-6 / self.dl) for temp in coordinate]
        return coordinate, coordinate_um, data
        
    def loadData(self,name=None):#读取加载电势场格点文件数据
        try:
            self.load_Settings_CustomBasis()
            print("加载自定义Basis设置")
        except Exception as er:
            print(er)
        if name is None:
            name = self.filename
        self.coordinate, self.coordinate_um, self.data = self.getData(name)
        self.x, self.y, self.z = self.coordinate

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
    与ism-cuda中的Configure类相同，用于统一电场参数读取方式

    :param basis: Data_Loader object containing electrode potential data and basis configuration
    :param V_static: dictionary of static voltages, key is electrode name, value is voltage in volts
    :param V_dynamic: dictionary of dynamic voltages, key is electrode name, value is amplitude in volts
    :param sym: whether to use symmetry
    :param g: cooling rate
    :param isotope_type: isotope type string
    """

    def __init__(self, basis, V_static: dict = {}, V_dynamic: dict = {}, sym: bool = False, g: float = 0.1, isotope_type: str = "Ba135") -> None:
        self.V_static = V_static
        self.V_dynamic = V_dynamic
        self.key_list = list(self.V_static.keys()) if self.V_static else []
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
        self.sym = sym
        self.g = g
        self.isotope_type = isotope_type

    def load_from_file(self, filename: str) -> None:
        """从JSON文件加载配置，格式与ism-cuda相同"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.V_dynamic = data.get("V_dynamic", {})
            if self.sym:
                # 对称性处理：将设置的电极电压分解为sym和asym两部分
                self.V_raw = data.get("V_static", {})
                key_ls = list(self.V_raw.keys())
                if "RF" in key_ls:
                    self.V_static["RF"] = self.V_raw["RF"]
                    key_ls.remove("RF")
                for key_id in range(len(key_ls)):
                    if key_id < len(key_ls) / 2 - 1:
                        self.V_static[key_ls[key_id]] = 0.5*(self.V_raw[key_ls[key_id]] + self.V_raw[key_ls[len(key_ls) - key_id - 1]])
                        self.V_static[key_ls[len(key_ls) - key_id - 1]] = 0.5*(self.V_raw[key_ls[key_id]] - self.V_raw[key_ls[len(key_ls) - key_id - 1]])
                    elif key_id == int((len(key_ls)) / 2):
                        self.V_static[key_ls[key_id]] = self.V_raw[key_ls[key_id]]
                        print("U8 is used")
                    else:
                        print(self.V_static)
                        break
            else:
                self.V_static = data.get("V_static", {})
        
        # 更新key_list
        self.key_list = list(self.V_static.keys())
        self.num_electordes = len(self.key_list)
    
    def load_from_param(self, V_static: dict, V_dynamic: dict) -> None:
        """从参数字典加载配置"""
        self.V_static = V_static
        self.V_dynamic = V_dynamic
        self.key_list = list(self.V_static.keys())
        self.num_electordes = len(self.key_list)

    def _interpret_voltage(self, value):
        """解释电压值，如果是列表则取第一个元素"""
        if type(value) == list:
            return value[0]
        return value

    def _interpret_dynamic(self, t):
        """解释动态电压的时间函数，默认使用cos(2*t)"""
        return np.cos(2*t)

    def _gen_grids(self, potential_static):
        """生成电场网格"""
        [x, y, z] = self.basis.coordinate
        fieldx, fieldy, fieldz = np.gradient(-potential_static, x, y, z, edge_order=2)
        grid_x = ionsim.Grid(x, y, z, value=fieldx)
        grid_y = ionsim.Grid(x, y, z, value=fieldy)
        grid_z = ionsim.Grid(x, y, z, value=fieldz)
        return [grid_x, grid_y, grid_z]

    def calc_field(self) -> None:
        """计算电场网格"""
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
        """计算力，与ism-cuda中的实现相同"""
        gamma = self.g
        bound_min = [np.min(self.basis.coordinate[i]) + 1e-9 for i in range(3)]
        bound_max = [np.max(self.basis.coordinate[i]) - 1e-9 for i in range(3)]
        mask = self.grids_dc[0].in_bounds(r)
        if len(np.where(mask == False)[0]) > 0:
            r = np.array([np.clip(r[:, i], bound_min[i], bound_max[i]) for i in range(3)]).T
            mask = self.grids_dc[0].in_bounds(r)
        r_mask = r[mask].copy(order='F')
        f = np.zeros_like(r)
        # inside bounds
        coord = self.grids_dc[0].get_coord(r_mask)
        f_in = (np.vstack(tuple([grid.interpolate(coord) for grid in self.grids_dc])))
        for key, value in self.grids_rf.items():
            grids_rf = value[0]
            f_in = f_in + np.vstack(tuple([grid.interpolate(coord) for grid in grids_rf])) * self._interpret_dynamic(t)
        f_in = f_in.transpose()
        f[mask] = f_in
        f = f - gamma * v
        return f

    def simulation(self, N: int, ini_range: int, mass: np.ndarray, charge: np.ndarray, step: int, interval: int, batch: int, t: float, device: bool, plotting: bool, alpha: float = 1.0, 
                   t_start: float = 0.0, config_name: str = "flat_28", save_final: bool = False, save_traj: bool = False, bilayer: bool = False, save_image_path: str = None) -> tuple:
        """运行模拟，与ism-cuda中的实现相同"""
        backend = CalculationBackend(device=device, step=step, interval=interval, batch=batch, time=t/(self.dt * 1e6), dt=self.dt, dl=self.dl, dV=self.dV, config_name=config_name, save_traj=save_traj, isotope_type=self.isotope_type, g=self.g)

        q1 = mp.Queue()
        q2 = mp.Queue(maxsize=50)

        status_dir = f"./data_cache/{N:d}/status/{config_name}/{self.isotope_type}/"

        if t_start > 0.1 and os.path.exists(status_dir+f"r/{t_start:.3f}us.npy"):
            r0 = np.load(status_dir+f"r/{t_start:.3f}us.npy")/(self.dl*1e6)    # In Status dir r is in the unit of um
            v0 = np.load(status_dir+f"v/{t_start:.3f}us.npy")/(self.dl/self.dt)   # In Status dir v is in the unit of m/s
            print("using stored data")
        else:
            r0 = (np.random.rand(N, 3)-0.5) *ini_range 
            v0 = np.zeros((N, 3))
            r0[:, 1] *= 0.1
            print("using y0.1 initial condition")
            if bilayer:
                r0[:N//2, 1] += 225/(self.dl*1e6)
                r0[N//2:, 1] -= 225/(self.dl*1e6)
        
        q1.put(Message(CommandType.START, r0, v0, t_start/(self.dt*1e6), mass, charge, self.calc_force))
        q2.put(Frame(r0, v0, t_start/(self.dt*1e6)))

        proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)   
        proc.start()

        f = None
        if plotting:
           # 启用实时绘图显示（与保存图片功能可以同时使用）
            show_plot = plotting
            plot = DataPlotter(q2, q1, Frame(r0, v0, t_start/(self.dt*1e6)), interval=0.04, z_range=200 if bilayer else 550, x_range=50 if bilayer else 100, y_range=250 if bilayer else 20, z_bias=0, dl=self.dl*1e6,dt=self.dt*1e6, bilayer=bilayer, mass=mass, show_plot=show_plot)
            f = plot.start(save_path=save_image_path)
            if not plot.is_alive():
                q1.put(Message(CommandType.STOP))
                while True:
                    new_f = q2.get()
                    if isinstance(new_f, bool) and not new_f:
                        break
                proc.join()
            std_y = f.r[:, 1].std() * self.dl * 1e6
        else:
            while True:
                new_f = q2.get()
                if isinstance(new_f, bool) and not new_f:
                    break
                f = new_f
            proc.join()
        
        if save_final:
            if not os.path.exists(status_dir+"r/"):
                os.makedirs(status_dir+"r/")
                os.makedirs(status_dir+"v/")
            np.save(status_dir + f"r/{f.timestamp*self.dt*1e6:.3f}us.npy", f.r*self.dl*1e6)
            np.save(status_dir + f"v/{f.timestamp*self.dt*1e6:.3f}us.npy", f.v*self.dl/self.dt)
        mask_final = (np.abs(f.r[:, 0]*self.dl*1e6)<100) & (np.abs(f.r[:, 1]*self.dl*1e6)<50) & (np.abs(f.r[:, 2]*self.dl*1e6)<1000)
        std_y = f.r[mask_final, 1].std()*self.dl*1e6
        len_z = np.abs(f.r[mask_final, 2].max() - f.r[mask_final, 2].min()) * self.dl * 1e6
        print("Lost ions: ", N-f.r[mask_final].shape[0])
        if bilayer:
            stdy_upper = (f.r[f.r[:,1]>0, 1]*self.dl*1e6).std()
            stdy_lower = (f.r[f.r[:,1]<0, 1]*self.dl*1e6).std()
            return stdy_upper, stdy_lower, len_z, f.timestamp * self.dt * 1e6
        else:
            return std_y, len_z, f.timestamp * self.dt * 1e6
