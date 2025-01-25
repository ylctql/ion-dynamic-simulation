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
# filename="./grid_data20241118.csv" #文件名：导入的电势场格点数据
filename="./grid_data20241118.csv"#ylc阱数据
#【电势场格点数据文件grid_data csv格式已更新！请务必参照文件的范例格式！】

loadGridsData_condition=[lambda x:x>=0,lambda y:y>=0,lambda z:z<=0] #对导入的电势场格点数据[x,y,z]坐标轴分别进行的条件筛选
# axis_symmetry=[0,1,2]#要对[x,y,z]中的哪几个轴进行对称化处理
axis_symmetry=[0,1,2]#ylc的阱中无对称操作

basis_filename="./electrode_basis.json"#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5
asymmetric_key="asy" # grid_data csv在对称化处理的时候，如果表头列名含有这个子字符串，那么按照反对称处理，否则默认按照对称

pi=math.pi
N = 300  #离子数
N_cur = N #仍然在阱中的离子数
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
def linear(t):
    return 2.5-min(t, 5)

V_static = {"RF":-6.4,"esbe4":[1, linear],"esbe3":-0.5}
# V_static = {"RF":-6.4,"esbe4":-0.5,"esbe2":0.0}# 静电场电压设置{"basis的文件名":该组电极施加电压(V)}
# V_static = {"U1": -5,"U2": 5,"U3": -5,"U4": 5,"U5": -5,"U6": 5,"U7": -5}# 静电场电压设置{"basis的文件名":该组电极施加电压(V)}
V_dynamic = {"RF":[Vrf,oscillate_RF]}
# V_dynamic = {"RF":[Vrf,oscillate_RF],"U4":[5, expramp],"U3":[-5, expramp], "U5":[-5, expramp], "U1":[-5, expramp]
#              , "U7":[-5, expramp], "U2":[5, expramp], "U6":[5, expramp]}# 含时动态电压设置{"basis的文件名":[该组电极施加电压(V),时间因子函数（最终相当于二者相乘）]}
# V_dynamic = {"RF":[Vrf,oscillate_RF]}# 含时动态电压设置{"basis的文件名":[该组电极施加电压(V),时间因子函数（最终相当于二者相乘）]}
# ,"esbe3":[-3,expramp]

        
epsl = 8.854*10**(-12)#真空介电常数@SI
m = 170.936*(1.66053904*10**(-27))#离子的质量 @SI #Yb171=170.936323；Yb174=173.938859
ec = 1.6*10**(-19)#元电荷@SI
#【normalization：在下面的程序求解过程中，“1”相当于什么呢？】
dt = 2/Omega#单位时间dt #dt*T*Omega=2pi ->  T=pi
dl = (( ec**2)/(4*pi*m*epsl*(Omega)**2))**(1/3)#单位长度dl
dV = m/ec*(dl/dt)**2 #单位电压dV
print('dt=',dt,' dl=',dl,' dV=',dV)

def mirror(data,ax,sign=1,exception=0):#相对于ax轴做镜面对称，ax=0 or 1 or 2。sign=1表示对称，sign=-1表示反对称
    #exception=0表示跳过对称面上的坐标零点
    data2=np.copy(data)
    data2[:,ax]*=-1
    if exception is not None:
        indices=        np.where(np.abs( data2[:,ax]-exception)>1e-9)[0]
        data2=data2[indices]
    data2[:,3:]*=np.array([sign])
    return data2
def symmetry(data,axis=None,sign=1):#相对于ax轴做镜面对称。sign=1表示对称，sign=-1表示反对称
    if axis is None:
        axis=[]
                
    for ax in axis:
        data=np.vstack((data,mirror(data,ax,sign,0)))
    return data
def cond_process(axi,cond):#工具函数：对坐标轴进行条件筛选
    if type(cond)==type(lambda x:x):
        # print(np.where(cond(axi)))
        return np.where(cond(axi))[0].tolist()
    else:

        #return np.where(axi==cond)[0].tolist()
        return np.where(axi>=0)[0].tolist()
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
        dat = dat[(loadGridsData_condition[0](dat[0]))  & (loadGridsData_condition[1](dat[1])) & (loadGridsData_condition[2](dat[2]))] #ylc:如果不用对称记得注释掉这行，如果要对称记得开这行
        data=dat.to_numpy()
        data[:,3:]*=1/dV#csv文件内的电势单位：V
        data[:,0:3]*=self.unit_l/dl #csv文件内坐标的长度单位：mm     
        data=symmetry(data=data,axis=axis_symmetry,sign=[(-1 if (asymmetric_key in key) else 1) for key in self.keynames])
        # 比如axis=[2] 表示关于z=0对称！且电势场名中含有asymmetric_key="asy"的按照反对称处理！
        #如果你只算了一个象限的数据，那就是axis=[0,1,2]
        print("data shape = ", data.shape)

        data = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]#index统一排序使得格点数据与文件内坐标顺序无关
        self.coordinate=[self.x, self.y, self.z] =[x,y,z]= [np.unique(data[:, i]) for i in range(3)]
        self.coordinate_um=[temp/(1e-6/dl) for temp in self.coordinate] 
        self.data = data 
        
    def load_basis(self,key):#加载电势场名key的格点数据
        coln=self.getcol(key)

        if coln is None:
            components=self.basisGroup_map[key]
            outputs=np.sum([self.load_basis(k)*v for k,v in components.items()],axis=0)
        else:
            outputs=self.data[:, coln].reshape(len(self.x), len(self.y), len(self.z))
        
        if flag_smoothing and self.smoothing is not None:
            outputs=self.smoothing(outputs)
        return outputs
    def load_Settings_CustomBasis(self):#加载自定义Basis设置
        self.basisGroup_map=loadConfig(basis_filename)
    def data_select(self,key,conditions=[True,True,True]):#选取电势场格点数据：电势场名key，conditions是对[x,y,z]坐标轴分别进行的条件筛选
        data=self.load_basis(key)
        a,b,c=(cond_process(self.coordinate_um[0],conditions[0]),cond_process(self.coordinate_um[1],conditions[1]),cond_process(self.coordinate_um[2],conditions[2]))
        return (data[a][:,b,:][:,:,c]),[self.coordinate_um[i][[a,b,c][i]] for i in range(3) ]
    def plotcurve(self,keys,conditions=[True,True,True],smoothfuns=None):#可视化：电势场数据画图
        result = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for key in keys:
            # if type(key)==type(np.array([])):
            data,coords=self.data_select(key,conditions)
            dims=np.array(list(np.shape(data)))
            aa=np.where(dims>1)[0]
            # if len(aa)==2:       
            # if len(aa)==3:       
                # print("3d")
            xx=X, Y = np.meshgrid(coords[aa[1]], coords[aa[0]])
            # 绘制 3D 曲面图
            # surf = ax.plot_surface(X, Y, np.squeeze(data), cmap='viridis')
            (size_x,size_y,size_z) = data.shape
            result += np.squeeze(data[:,:,int(size_z/2)+1])
        
        surf = ax.plot_surface(X, Y, result, cmap='viridis')
        plt.show()
        #     else:
        #         for i in range(3) :
        #             if len(coords[i])>1:
        #                 xx=coords[i]
        #                 plt.plot(coords[i],data.flatten(),"-o",label="comsol")
        #                 if smoothfuns is not None:
        #                     for fun in smoothfuns:
        #                         plt.plot(coords[i],fun(coords[i],data.flatten()),"-o")
        #                 plt.legend()
        #                 plt.show()
        # # return [xx,data.flatten()]
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

# 考虑对单个电极的对称和反对称分解才需要考虑
def basis_preset():#工具函数
    data_loader.basisGroup_map={"RF":{"esbe6":1},"Ucenter":{"esbe4":1},"U4":{"esbe4":1}}
    data_loader.basisGroup_map["U1"]={"esbe1":1/2,"esbe1asy":-1/2}
    data_loader.basisGroup_map["U2"]={"esbe2":1/2,"esbe2asy":-1/2}
    data_loader.basisGroup_map["U3"]={"esbe3":1/2,"esbe3asy":-1/2}
    data_loader.basisGroup_map["U7"]={"esbe1":1/2,"esbe1asy":1/2}
    data_loader.basisGroup_map["U6"]={"esbe2":1/2,"esbe2asy":1/2}
    data_loader.basisGroup_map["U5"]={"esbe3":1/2,"esbe3asy":1/2}
    data_loader.basisGroup_map["Unear"]={"esbe1":3}
    saveConfig(data_loader.basisGroup_map,basis_filename)
    exit()    


def force(r: np.ndarray, v: np.ndarray, t: float):
    gamma =0.05# (0.7 - t / 30 * 0.6) if t < 30 else 0.05#阻尼因子，代表Doppler cooling强度
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
        fun_=value[1]   #此处的fun_代表余弦振荡的函数
        f_in=f_in+  np.vstack(tuple([grid.interpolate(coord) for grid in grids_rf] ))*interpret_dynamic(fun_,t)#加上含时的力
    f_in=f_in.transpose()
    f[mask] =f_in
    # outside bounds# r_nmask = r[~mask].copy(order='F')# f[~mask] = np.zeros_like(r_nmask)#一般来说这里为空
    f=f-gamma*v#Doppler cooling
    return f
data_loader=Data_Loader(filename)
data_loader.loadData()
# data_loader.plotcurve(["esbe1", "esbe2", "esbe3", "esbe4"])#改成可以看总电场了

# 加载静电势场
potential_static=0
for key,value in V_static.items():
    potential_static=potential_static+data_loader.load_basis(key)*interpret_voltage(value) #静电场直接叠加
data_loader.grids_dc=gen_grids(potential_static)

#加载含时动态电势场
potential_dynamic_list=[]
grids_dynamic_dict={}
for key,value in V_dynamic.items():
    potential_dynamic_list.append(data_loader.load_basis(key)*interpret_voltage(value)) #此处的value为列表，内容上为[275（数值）, cos(2t)（函数）]
    grids_dynamic_dict[key]=[gen_grids(potential_dynamic_list[-1]),value] #含时的分开处理，时间因子不一样

bound_min=[np.min(data_loader.coordinate[i])+1e-9 for i in range(3)]
bound_max=[np.max(data_loader.coordinate[i])-1e-9 for i in range(3)]
print("bound_min=",bound_min,"bound_max=",bound_max)#网格边界

if __name__ == "__main__":

    
    backend = CalculationBackend(step=10, interval=1, batch=50)#step越大精度越高
    ini_range=100#影响画图范围和初始离子坐标
    r0 = (np.random.rand(N, 3)-0.5) *ini_range
    try:    r0=np.loadtxt("./traj/r.txt")
    except:pass
    
    v0 = np.zeros((N, 3))

    q1 = mp.Queue()
    q2 = mp.Queue(maxsize=50)

    q1.put(Message(CommandType.START, r0, v0, mass, charge, force))
    q2.put(Frame(r0, v0, 0))

    plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.04,ini_range=ini_range,dl=dl*1e6,dt=dt*1e6)
    proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)
    
    proc.start()
    plot.start(plotFlag=1)#True表示只画图 #False表示只输出轨迹到/traj/文件夹# "both"表示二者都做

    if proc.is_alive():
        q1.put(Message(CommandType.STOP))
        while True:
            f = q2.get()
            if isinstance(f, bool) and not f:
                break
        proc.join()