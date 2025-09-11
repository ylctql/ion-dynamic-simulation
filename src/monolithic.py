import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   
import ionsim
from utils import *
from dataplot import *    
import math,csv,traceback,json
from scipy.signal import savgol_filter
flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
filename="../data/monolithic20241118.csv" #文件名：导入的电势场格点数据
basis_filename="./electrode_basis.json"#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5

pi=math.pi
N = 150  #离子数
charge = np.ones(N) #每个离子带电荷量都是1个元电荷
mass = np.ones(N) #每个离子质量都是1m，具体大小见下面的m
Vrf=550/2 #RF电压振幅
freq_RF=35.28 #RF射频频率@MHz
Omega = freq_RF*2*pi*10**6 #RF射频角频率@SI

def oscillate_RF(t):#RF场的时间函数。单位时间为dt
    # return np.cos(Omega*t*1e-6*dt)
    return np.cos(2*t)
def pushDC(t):#axial shuttling场的时间函数。单位时间为dt
    return np.cos(2*t/50)
def expdecay(t):#时间函数：指数衰减
    return np.exp(-t/50)
def expramp(t):#时间函数
    return 1-expdecay(t)
def cut(t):
    if t*dt*1e6<1:
        return 1
    else:
        return 0

V_static = {"RF":-6.4, "U4":-0.5}
V_dynamic = {"RF":[Vrf,oscillate_RF]}# 含时动态电压设置{"basis的文件名":[该组电极施加电压(V),时间因子函数（最终相当于二者相乘）]}

epsl = 8.854*10**(-12)#真空介电常数@SI
m = 170.936*(1.66053904*10**(-27))#离子的质量 @SI #Yb171=170.936323；Yb174=173.938859
ec = 1.6*10**(-19)#元电荷@SI
#【normalization：在下面的程序求解过程中，“1”相当于什么呢？】
dt = 2/Omega#单位时间dt #dt*T*Omega=2pi ->  T=pi
dl = (( ec**2)/(4*pi*m*epsl*(Omega)**2))**(1/3)#单位长度dl
dV = m/ec*(dl/dt)**2 #单位电压dV
print('dt=',dt,' dl=',dl,' dV=',dV)

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
    def plotcurve(self,key,conditions=[True,True,True],smoothfuns=None):#可视化：电势场数据画图
        # if type(key)==type(np.array([])):
        data,coords=self.data_select(key,conditions)
        dims=np.array(list(np.shape(data)))
        aa=np.where(dims>1)[0]
        if len(aa)==2:        
            xx=X, Y = np.meshgrid(coords[aa[1]], coords[aa[0]])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # 绘制 3D 曲面图
            surf = ax.plot_surface(X, Y, np.squeeze(data), cmap='viridis')
            plt.show()
        else:
            for i    in range(3) :

                if len(coords[i])>1:
                    xx=coords[i]
                    plt.plot(coords[i],data.flatten(),"-o",label="comsol")
                    if smoothfuns is not None:
                        for fun in smoothfuns:
                            plt.plot(coords[i],fun(coords[i],data.flatten()),"-o")
                    plt.legend()
                    plt.show()
        return [xx,data.flatten()]
def interpret_voltage(value):#工具函数
    if type(value)==list:
        return  value[0]
    return value
def interpret_dynamic(value,t):#工具函数
    if type(value)==list:
        for i in range(len(value)):
            if type(value[i])==type(interpret_voltage):
                return value[i](t,*[value[j] for j in range(i+1,len(value))])
    return value
def gen_grids(potential_static):#工具函数
    [x,y,z]=data_loader.coordinate
    fieldx, fieldy, fieldz = np.gradient(-potential_static, x, y, z, edge_order=2)
    grid_x = ionsim.Grid(x, y, z, value=fieldx)
    grid_y = ionsim.Grid(x, y, z, value=fieldy)
    grid_z = ionsim.Grid(x, y, z, value=fieldz)        
    return [grid_x,grid_y,grid_z]
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

def force(r: np.ndarray, v: np.ndarray, t: float):
    gamma = 0.1

    f = np.zeros_like(r)
    mask =  data_loader.grids_dc[0].in_bounds(r)#大家都没出界吧？
    if len( np.where(mask==False)[0] )>0:
        print("警告出界")    
        r = np.array([np.clip(r[:,i],bound_min[i],bound_max[i]) for i in range(3)]).T#如果出界，就按边界上的电场
        mask =  data_loader.grids_dc[0].in_bounds(r)
    r_mask = r[mask].copy(order='F')
    

    # inside bounds
    coord = data_loader.grids_dc[0].get_coord(r_mask)
    f_in = (np.vstack(tuple([grid.interpolate(coord) for grid in data_loader.grids_dc] ))       )#静电力
    for key,value  in grids_dynamic_dict.items():
        grids_rf=value[0]
        fun_=value[1]
        f_in=f_in+  np.vstack(tuple([grid.interpolate(coord) for grid in grids_rf] ))*interpret_dynamic(fun_,t)#加上含时的力
    f_in=f_in.transpose()
    f[mask] =f_in
    # outside bounds# r_nmask = r[~mask].copy(order='F')# f[~mask] = np.zeros_like(r_nmask)#一般来说这里为空
    f=f-gamma*v#Doppler cooling
    return f
data_loader=Data_Loader(filename)
data_loader.loadData()

# 加载静电势场
potential_static=0
for key,value in V_static.items():
    potential_static=potential_static+data_loader.load_basis(key)*interpret_voltage(value) #静电场直接叠加
data_loader.grids_dc=gen_grids(potential_static)

#加载含时动态电势场
potential_dynamic_list=[]
grids_dynamic_dict={}
for key,value in V_dynamic.items():
    potential_dynamic_list.append(data_loader.load_basis(key)*interpret_voltage(value))
    grids_dynamic_dict[key]=[gen_grids(potential_dynamic_list[-1]),value] #含时的分开处理，时间因子不一样

# 绘制赝势
def psudo_potential():
    V0 = data_loader.load_basis("RF")*interpret_voltage(Vrf)*ec*dV #此处统一使用国际单位制
    [x, y, z] = data_loader.coordinate_um   #换算成国际单位制
    Fx, Fy, Fz = np.gradient(-V0, x, y, z, edge_order=2)   
    F0 = np.sqrt(Fx**2 + Fy**2 + Fz**2)*1e6 #因为距离单位为um，所以梯度要乘1e6
    V_pseudo_rf = F0**2/(4*m*Omega**2*ec)
    Vs = potential_static*dV
    shape = Vs.shape
    Vs_y0 = Vs[:, int(np.ceil(shape[1]/2)), :]
    Vp_y0 = V_pseudo_rf[:, int(np.ceil(shape[1]/2)), :]
    X, Y = np.meshgrid(z, x)   
    fig = plt.figure()
    # plt.rcParams.update({'font.size': 14})
    # ax_s = fig.add_subplot(111, projection='3d')
    # ax_s.plot_surface(X,Y,Vs_y0)
    # ax_s.set_xlabel('Z(um)', fontsize=20)
    # ax_s.set_ylabel('X(um)', fontsize=20)
    # ax_s.set_zlabel('Vs(V)', fontsize=20)
    # ax_s.tick_params(labelsize=20)
    # ax_s.xaxis.labelpad = 20  # 调整 X 轴标题与轴的距离
    # ax_s.yaxis.labelpad = 20  # 调整 Y 轴标题与轴的距离
    # ax_s.zaxis.labelpad = 20  # 调整 Z 轴标题与轴的距离
    # ax_s.set_xticklabels(ax_s.get_xticks(), fontsize=12)
    # ax_s.set_yticklabels(ax_s.get_yticks(), fontsize=12)
    # ax_s.set_zticklabels(ax_s.get_zticks(), fontsize=12)
    # ax_s.zaxis.label.set_rotation(90)
    
    # ax_p = fig.add_subplot(122, projection='3d')
    # ax_p.plot_surface(X,Y,Vp_y0)
    # ax_p.set_xlabel('Z(um)', fontsize=14)
    # ax_p.set_ylabel('X(um)', fontsize=14)
    # ax_p.set_zlabel('Vpp(V)', fontsize=14)
    # ax_p.set_xticklabels(ax_p.get_xticks(), fontsize=12)
    # ax_p.set_yticklabels(ax_p.get_yticks(), fontsize=12)
    # ax_p.set_zticklabels(ax_p.get_zticks(), fontsize=12)
    ax_p = fig.add_subplot(111, projection='3d')
    ax_p.plot_surface(X,Y,Vp_y0)
    ax_p.set_xlabel('Z(um)', fontsize=20)
    ax_p.set_ylabel('X(um)', fontsize=20)
    ax_p.set_zlabel('Vpp(V)', fontsize=20)
    ax_p.tick_params(labelsize=20)
    ax_p.xaxis.labelpad = 20  # 调整 X 轴标题与轴的距离
    ax_p.yaxis.labelpad = 20  # 调整 Y 轴标题与轴的距离
    ax_p.zaxis.labelpad = 20  # 调整 Z 轴标题与轴的距离
    plt.show()

def save_potential():
    V0 = data_loader.load_basis("RF")*interpret_voltage(Vrf)*ec*dV #此处统一使用国际单位制
    [x, y, z] = data_loader.coordinate_um   #换算成国际单位制
    Fx, Fy, Fz = np.gradient(-V0, x, y, z, edge_order=2)   
    F0 = np.sqrt(Fx**2 + Fy**2 + Fz**2)*1e6 #因为距离单位为um，所以梯度要乘1e6
    V_pseudo_rf = F0**2/(4*m*Omega**2*ec)
    Vs = potential_static*dV
    V = V_pseudo_rf+Vs
    np.save("./potential/grid_x.npy", x)
    np.save("./potential/grid_y.npy", y)
    np.save("./potential/grid_z.npy", z)
    np.save("./potential/300DC50gap_zlarger.npy", V)

bound_min=[np.min(data_loader.coordinate[i])+1e-9 for i in range(3)]
bound_max=[np.max(data_loader.coordinate[i])-1e-9 for i in range(3)]
print(bound_min,bound_max)#网格边界

if __name__ == "__main__":

    # backend = CalculationBackend(step=100, interval=5, batch=50)#step越大精度越高
    backend = CalculationBackend(step=10, interval=5, batch=50)
    # ini_range=100#影响画图范围和初始离子坐标
    ini_range = np.random.randint(100, 200) #初始范围也随机，探索更多可能

    # r0 = r0[np.lexsort([r0[:, 1], r0[:, 0], r0[:, 2]])]  #依次按z-x-y从小到大排序
    # r0 = np.loadtxt("./balance/balance.txt")/(1e6*dl) #从平衡位置开始演化
    r0 = (np.random.rand(N, 3)-0.5) *ini_range
    v0 = np.zeros((N, 3))

    q1 = mp.Queue()
    q2 = mp.Queue(maxsize=50)

    q1.put(Message(CommandType.START, r0, v0, mass, charge, force))
    q2.put(Frame(r0, v0, 0))

    plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.04, z_range=100, z_bias=0, dl=dl*1e6,dt=dt*1e6)
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