import taichi as ti
from taichi import math as tm
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import random
import pandas as pd


# Choose any of the following backend when initializing Taichi
# - ti.cpu
# - ti.gpu
# - ti.cuda
# - ti.vulkan
# - ti.metal
# - ti.opengl

def byBackendmainList(backend, N: int, rowOfVars: np.ndarray):
    assert rowVars.shape[0] == N
    ti.init(arch=backend)
    pixels = ti.field(dtype=float, shape=rowOfVars.shape)
    pixels.from_numpy(rowOfVars)
    # print(f'pixels {pixels}')

    @ti.kernel
    def sinParallel(pixels: ti.template()):
        for i, j in pixels:
            pixels[i, j] = tm.sin(pixels[i, j])

    start = time.time()
    sinParallel(pixels)
    timeAns = time.time() - start

    ti.reset()
    return pixels, timeAns


ex = []
times = [[], [], []]
names = ["Vulkan", "CPU", 'numpy']
for numb in [10 ** i for i in range(4, 7)]:
    rowVars = np.random.uniform(-10, 10, size=(numb, 1))
    _, vulkanTime = byBackendmainList(ti.vulkan, N=numb, rowOfVars=rowVars)
    _, cpuTime = byBackendmainList(ti.cpu, N=numb, rowOfVars=rowVars)

    startNP = time.time()
    npSin = np.sin(rowVars)
    timeNP = time.time() - startNP

    print(f"\nvulkanTime: {vulkanTime} sec")
    print(f"cpuTime: {cpuTime} sec\n")

    ex.append(numb)
    times[0].append(vulkanTime)
    times[1].append(cpuTime)
    times[2].append(timeNP)

######## PLOT ########
df = pd.DataFrame({
    'Vulkan': times[0],
    'CPU': times[1],
    "numpy": times[2]
})
df.plot(kind='line')
plt.show()
