import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve, generate_binary_structure

# Gray-Scott模型参数(参见理论引用：https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/)
F = 0.04
k = 0.06075
Du, Dv = 0.16, 0.08
# 设定网格大小
size = 20
U = np.ones((size, size))
V = np.zeros((size, size))
# 初始条件：在中心放置一小块V
mid = size // 2
r = 2
U[mid-r:mid+r, mid-r:mid+r] = 0.50
V[mid-r:mid+r, mid-r:mid+r] = 0.25

# 用于计算拉普拉斯算子的结构元素
s = generate_binary_structure(2, 1)

# 反应扩散模型的迭代步骤
def reaction_diffusion_step(U, V, Du, Dv, F, k, dt):
    Lu = convolve(U, s, mode='reflect') - U
    Lv = convolve(V, s, mode='reflect') - V
    UVV = U * V**2
    U += (Du * Lu - UVV + F * (1 - U)) * dt
    V += (Dv * Lv + UVV - (F + k) * V) * dt
    return U, V

# 初始化图形
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# 初始化体素图形
def init():
    X, Y, Z = np.indices((size, size, size))
    cube = (X >= 0) & (Y >= 0) & (Z >= 0)
    voxels = ax.voxels(cube, facecolors='blue', edgecolor='k', shade=False)
    return voxels,

def update(frame):
    global U, V
    U, V = reaction_diffusion_step(U, V, Du, Dv, F, k, 1.0)
    # 清除前一帧的体素
    ax.collections.clear()
    # 创建一个新的体素图
    ax.voxels(V > 0.1, facecolors='blue', edgecolor='k', shade=False)

anim = FuncAnimation(fig, update, frames=100, init_func=init, blit=False)

plt.show()
