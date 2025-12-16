# Sequence analyse
import numpy as np
import os
from configure import *
from matplotlib.ticker import MultipleLocator

dirname = os.getcwd()
flag_smoothing=True #是否对导入的电势场格点数据做平滑化，如果True，那么平滑化函数默认按照下文def smoothing(data)
filename=os.path.join(dirname, "../../../data/monolithic20241118.csv") #文件名：导入的电势场格点数据
basis_filename=os.path.join(dirname, "electrode_basis.json")#文件名：自定义Basis设置 #可以理解为一种基矢变换，比如"U1"相当于电势场组合"esbe1"*0.5+"esbe1asy"*-0.5
basis = Data_Loader(filename, basis_filename, flag_smoothing)
basis.loadData()
configure = Configure(basis=basis)

dir_path = "../data_cache/ion_pos/10000/"
items = os.listdir(dir_path)
# 仅统计文件（排除子文件夹）
files = [item for item in items if os.path.isfile(os.path.join(dir_path, item))]
Nf = len(files)
print(Nf)
Dt = configure.dt*1e6
seq = np.zeros(Nf//2*2)
Nth = 5000
ax_ion = 2
for idx in range(Nf):
    timestamp = idx//2
    if idx//2 == 0:
        real_t = timestamp
    else:
        real_t = timestamp+0.5
    _time = real_t*Dt
    # consider 0th ion
    try:
        r = np.load(dir_path+"timestamp%d&time%.3fus.npy"%(timestamp, _time))
    except:
        break
    seq[idx] = r[Nth, ax_ion]    # Nth ion's motion on ax_ion's direction
t_seq = Dt/2*np.arange(Nf)

fs = 2/Dt # Sampling frequency
freq = np.fft.fft(seq)
Ns = len(freq)
freqs = np.fft.fftfreq(Ns, 1/fs)  # 频率轴
magnitude = np.abs(freq) / Ns  # 幅度谱（归一化）

# 只绘制正频率部分
half_Ns = Ns // 2
freqs = freqs[:half_Ns]
magnitude = magnitude[:half_Ns]

# 绘制时域信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t_seq, seq)
plt.title("Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# 绘制频域幅度谱
plt.subplot(2, 1, 2)
plt.stem(freqs, magnitude, basefmt=" ")
plt.title("Freq spectral")
plt.xlabel("Freq (MHz)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(0, 100, 10))
# plt.xticks(np.arange(-100, 110, 10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.grid(True, which='major', linestyle='-', color='gray', alpha=0.7)  # 主刻度网格
plt.grid(True, which='minor', linestyle=':', color='lightgray', alpha=0.5)  # 次刻度网格

plt.tight_layout()
plt.show()

