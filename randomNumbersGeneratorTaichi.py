import  taichi as ti
from  taichi import math as tm
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import random
import  pandas as pd

# Choose any of the following backend when initializing Taichi
# - ti.cpu
# - ti.gpu
# - ti.cuda
# - ti.vulkan
# - ti.metal
# - ti.opengl

def byBackendmainList(backend, N : int):

    ti.init(arch=backend)
    pixels = ti.field(dtype=float, shape=(N,))

    @ti.kernel
    def generateRandomList():
        for i in pixels:
            pixels[i] = ti.random(dtype=float)


    start = time.time()
    generateRandomList()
    timeAns = time.time() - start

    ti.reset()
    return pixels,timeAns


vulkanList,vulkanTime = byBackendmainList(ti.vulkan,1000)
cpuList,cpuTime = byBackendmainList(ti.cpu,1000)
print('\nANSWERS\n')

print(f"vulkanTime: {vulkanTime} sec")
print(f"cpuTime: {cpuTime} sec\n")

#print(f"1 cpuList: {cpuList} ")
#print(f"\n2 vulkanList: {vulkanList} ")

colors_list = list(colors._colors_full_map.values())[:3]

ex = []
times = [[],[]]
names = ["Vulkan","CPU"]
for numb in [10**i for i in range(5,10)]:
    _, vulkanTime = byBackendmainList(ti.vulkan, numb)
    _, cpuTime = byBackendmainList(ti.cpu, numb)

    print(f"\nvulkanTime: {vulkanTime} sec")
    print(f"cpuTime: {cpuTime} sec\n")

    ex.append(numb)
    times[0].append(vulkanTime)
    times[1].append(cpuTime)

######## PLOT ########
df = pd.DataFrame({'Vulkan': times[0], 'CPU': times[1]})
df.plot(kind='line')
plt.show()