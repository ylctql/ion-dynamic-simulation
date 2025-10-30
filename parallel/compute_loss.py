import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   
import ionsim
from utils import *
from dataplot import *    
import math,csv,traceback,json
from scipy.signal import savgol_filter
import os
import json
from functools import partial
from scipy.interpolate import RegularGridInterpolator
import argparse

dirname = os.path.dirname(__file__)
flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
filename=os.path.join(dirname, "../../../data/60electrodes_tiny.csv")
basis_filename=os.path.join(dirname, "electrode_basis.json")#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5

pi=math.pi
Vrf=550/2 #RF电压振幅
freq_RF=35.28 #RF射频频率@MHz
Omega = freq_RF*2*pi*10**6 #RF射频角频率@SI
epsl = 8.854*10**(-12)#真空介电常数@SI
m = 170.936*(1.66053904*10**(-27))#离子的质量 @SI #Yb171=170.936323；Yb174=173.938859
ec = 1.6*10**(-19)#元电荷@SI
#【normalization：在下面的程序求解过程中，“1”相当于什么呢？】
dt = 2/Omega#单位时间dt #dt*T*Omega=2pi ->  T=pi
dl = (( ec**2)/(4*pi*m*epsl*(Omega)**2))**(1/3)#单位长度dl
dV = m/ec*(dl/dt)**2 #单位电压dV
print('dt=',dt,' dl=',dl,' dV=',dV)

N = 1000  #离子数
charge = np.ones(N) #每个离子带电荷量都是1个元电荷
mass = np.ones(N) #每个离子质量都是1m，具体大小见下面的m


def oscillate_RF(t):#RF场的时间函数。单位时间为dt
    # return np.cos(Omega*t*1e-6*dt)
    return np.cos(2*t)

V_dynamic = {"RF":Vrf}# 含时动态电压设置{"basis的文件名":[该组电极施加电压(V),时间因子函数（最终相当于二者相乘）]}

def smoothing(data):#由于电势场数据崎岖不平，因而做的平滑化函数
    return savgol_filter(data,11,3)  
class Data_Loader:
    def __init__(self,filename=filename) -> None:
        self.filename=filename
        self.data=[]
        self.keymaps={}
        self.keynames=[]
        self.basis={}
        self.unit_l=1e-3
        self.basisGroup_map={}
        self.units={"m":1e0,"cm":1e-2,"mm":1e-3,"um":1e-6}
        self.smoothing=smoothing
        self.grids_dc=None
        self.grids_rf=None
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
            name=self.filename
        with open(name,encoding='utf-8',mode='r+') as file_read:
            csvread = csv.reader(file_read)
            for i,row in enumerate(csvread):
                if i>20:break
                if row[0]==r'% Length unit':
                     self.unit_l=self.units[row[1]]
                     print("self.unit_l=",self.unit_l)                  
                if row[0]==r'% x':
                    self.keymaps={row[v].replace(r"% ",""):v for v in range(len(row))}
                    self.keynames=[name for name in self.keymaps if  name not in ["x","y","z"]]
                    break
        dat = pd.read_csv(name, comment='%', header=None)
        data=dat.to_numpy()
        data[:,3:]*=1/dV#csv文件内的电势单位：V
        data[:,0:3]*=self.unit_l/dl #csv文件内坐标的长度单位：mm     

        data = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]#index统一排序使得格点数据与文件内坐标顺序无关
        self.coordinate=[self.x, self.y, self.z] =[x,y,z]= [np.unique(data[:, i]) for i in range(3)]
        self.coordinate_um=[temp/(1e-6/dl) for temp in self.coordinate] 
        self.data = data 
        
    def load_basis(self,key):#加载电势场名key的格点数据
        coln=self.getcol(key)

        if coln is None:
            components=self.basisGroup_map[key]
            outputs = self.load_basis(components)
        else:
            outputs=self.data[:, coln].reshape(len(self.x), len(self.y), len(self.z))
        
        if flag_smoothing and self.smoothing is not None:
            outputs=self.smoothing(outputs)
        return outputs
    def load_Settings_CustomBasis(self):#加载自定义Basis设置
        self.basisGroup_map=loadConfig(basis_filename)

def loadConfig(fileName):#工具函数
    try:
        with open(fileName, 'r',encoding='utf-8') as jf:
            config = json.load(jf)
        return config
    except Exception as er:
        # print("无法加载配置文件",fileName,er)         traceback.print_stack()
        print(traceback.format_exc())
        return -1

def saveConfig(config, fileName,):#工具函数
    with open(fileName, mode="w+",encoding='utf-8') as jf:
        jf.write(json.dumps(config,indent=4, ensure_ascii=False))

data_loader=Data_Loader(filename)
data_loader.loadData()

with open("electrode_basis.json", 'r',encoding='utf-8') as jf:
    basis_dict = json.load(jf)

# The basis would just be loaded once
electorde_basis = {}
for key in basis_dict.keys():
    electorde_basis[key]=data_loader.load_basis(key)
for key in V_dynamic.keys():
    if key not in electorde_basis:
        electorde_basis[key]=data_loader.load_basis(key)

[grid_x, grid_y, grid_z] = data_loader.coordinate
# def gen_grids(potential_static):#工具函数
#     fieldx, fieldy, fieldz = np.gradient(-potential_static, grid_x, grid_y, grid_z, edge_order=2)
#     # 矢量场的三个分量对应三个标量场
#     grid_x = ionsim.Grid(grid_x, grid_y, grid_z, value=fieldx)
#     grid_y = ionsim.Grid(grid_x, grid_y, grid_z, value=fieldy)
#     grid_z = ionsim.Grid(grid_x, grid_y, grid_z, value=fieldz)        
#     return [grid_x,grid_y,grid_z]
def compute_field(potential_static):
    fieldx, fieldy, fieldz = np.gradient(-potential_static, grid_x, grid_y, grid_z, edge_order=2)
    return [fieldx, fieldy, fieldz]

# The initial positions and velocities should only be set once
# backend = CalculationBackend(step=100, interval=5, batch=50)#step越大精度越高
backend = CalculationBackend(step=10, interval=5, batch=50)
np.random.seed(42)
# ini_range=100#影响画图范围和初始离子坐标
ini_xrange = 150
ini_zrange = 300 
ini_yrange = 70 # y方向Grid_size较小，防止出界
r0 = (np.random.rand(N, 3)-0.5)
r0[:,0] *= ini_xrange
r0[:,1] *= ini_yrange
r0[:,2] *= ini_zrange
v0 = np.zeros((N, 3))

#加载含时动态电势场，默认全部按照oscillate_RF(t)函数变化
potential_dynamic=0
for key,value in V_dynamic.items():
    potential_dynamic+=electorde_basis[key]*value
grid_rf_fun = [RegularGridInterpolator((grid_x, grid_y, grid_z), compute_field(potential_dynamic)[dim]) for dim in range(3)]
# dynamic_grid = gen_grids(potential_dynamic)

bound_min=[np.min(data_loader.coordinate[i])+1e-9 for i in range(3)]
bound_max=[np.max(data_loader.coordinate[i])-1e-9 for i in range(3)]
print(bound_min,bound_max)#网格边界

def compute_force(r: np.ndarray, v: np.ndarray, t: float, grid_dc_fun):
    gamma = 0.1

    f = np.zeros_like(r)

    # mask =  grid_dc[0].in_bounds(r)#大家都没出界吧？
    # print("Max range:", np.max(r, axis=0))
    mask = np.all((r>=bound_min) & (r<=bound_max),axis=1)
    if len(np.where(mask==False)[0])>0:
        print("警告出界")    
        r = np.array([np.clip(r[:,i],bound_min[i],bound_max[i]) for i in range(3)]).T#如果出界，就按边界上的电场
        mask =  np.all((r>=bound_min) & (r<=bound_max),axis=1)
    r_mask = r[mask].copy(order='F')
    # r_mask = r
    # inside bounds
    # coord = grid_dc[0].get_coord(r_mask)
    f_in = (np.vstack(tuple([grid_fun(r_mask) for grid_fun in grid_dc_fun])))#静电力
    f_in += np.vstack(tuple([grid_fun(r_mask) for grid_fun in grid_rf_fun]))*oscillate_RF(t)#加上含时的力
    f_in=f_in.transpose()
    f[mask] =f_in
    # f = f_in
    # outside bounds# r_nmask = r[~mask].copy(order='F')# f[~mask] = np.zeros_like(r_nmask)#一般来说这里为空
    f=f-gamma*v#Doppler cooling
    # f = np.zeros_like(r)
    # print("forcetype:",type(f),f.shape)
    return f

def R2(x,y):
    '''
    x: fit data
    y: raw data
    '''
    return 1-np.sum((x-y)**2)/np.sum((y-np.mean(y))**2)

class compute_loss():
    def __init__(self, U, h):
        self.U = U
        self.h = h
        '''
        U: The voltage list to be varied
        h: The perturbation for finite difference
        '''
        self.V_static = {}
        key_id = 0
        for key in basis_dict.keys():
            if key_id < len(self.U):
                self.V_static[key]=self.U[key_id]
            else:
                self.V_static[key]=self.U[len(basis_dict.keys())-key_id]    #对称电极施加相同电压
            key_id += 1
        
        # 加载静电势场
        potential_static=0
        for key,value in self.V_static.items():
            potential_static+=electorde_basis[key]*value #静电场直接叠加
        
        # 由于电压变化产生的静电势变化
        # self.grid_dc=[gen_grids(potential_static+self.h*electorde_basis[key]) for key in basis_dict.keys()]
        self.grid_dc_fun = [[RegularGridInterpolator((grid_x, grid_y, grid_z), compute_field(potential_static + self.h * electorde_basis[key])[dim]) for dim in range(3)] for key in basis_dict.keys()]

    def loss(self):
        loss = np.zeros(len(self.U))
        for i in range(len(self.U)):
            q1 = mp.Queue()
            q2 = mp.Queue(maxsize=50)

            q1.put(Message(CommandType.START, r0, v0, charge, mass, partial(compute_force, grid_dc_fun=self.grid_dc_fun[i])))
            q2.put(Frame(r0, v0, 0))

            plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.04, z_range=350, x_range=50, z_bias=0, y_range=5, dl=dl*1e6,dt=dt*1e6)
            proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)
            
            proc.start()
            plot.start()#True表示只画图 #False表示只输出轨迹到/traj/文件夹# "both"表示二者都做

            if proc.is_alive():
                q1.put(Message(CommandType.STOP))
                while True:
                    f = q2.get()
                    if isinstance(f, bool) and not f:
                        break
                proc.join()
            if self.h < 0.001: #h=0表示不是为了算梯度，而是只返回loss
                print("Finished computing loss for U =", self.U)
                return plot.last_loss
            loss[i] = plot.last_loss
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute loss for given electrode voltages')
    parser.add_argument('--U', nargs='+', type=float, default=[-6.4]+8*[0], help='Voltages on electrodes')
    args = parser.parse_args()
    U = args.U
    loss_computer = compute_loss(U, h=0)
    L = loss_computer.loss()
    print("Loss:", L)



    
    
    