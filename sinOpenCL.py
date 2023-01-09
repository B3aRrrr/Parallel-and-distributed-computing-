#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import time
from matplotlib import pyplot as plt


def OpenCLCalcSIN(a_matrix: np.ndarray, number: int, a_np: np.ndarray) -> (np.ndarray, time.time()):
    start = time.time()

    # context = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_matrix)

    prg = cl.Program(ctx, """
    __kernel void mySin(
        __global  float * data)
    {
          size_t id  = get_global_id(1) * get_global_size(0) + get_global_id(0);
          data[id] = sin(data[id]);
        }
    """).build()

    data = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    knl = prg.mySin  # Use this Kernel object for repeated calls
    knl(queue, a_np.shape, None, data)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, data)

    return res_np, time.time() - start


def numpyCalcSIN(a_np: np.ndarray) -> (np.ndarray, time.time()):
    start = time.time()
    return np.sin(a_np), time.time() - start


"""###TESTS###"""
timeListGPU = []
timeListCPU = []

nExamples = []
# from matplotlib.pyplot import figure
# figure(figsize=(8, 6), dpi=80)
for i in range(100000, 1100000, 50000):
    a_np = np.random.rand(i).astype(np.float32)
    #
    # number = i
    N = int(i ** 0.5) + 1 * np.sign(int(i - int(i ** 0.5) * int(i ** 0.5)))

    a_matrix = np.zeros((N, N))
    for i in range(len(a_np)):
        a_matrix[i // N][i - (i // N) * N] = a_np[i]
    a_matrix = a_matrix.astype(np.float32)

    nExamples.append(i)

    resGPU, timeGPU = OpenCLCalcSIN(a_matrix, i, a_np)
    timeListGPU.append(timeGPU)

    resCPUnumpy, timeCPUnumpy = numpyCalcSIN(a_np)
    timeListCPU.append(timeCPUnumpy)

n = np.array(nExamples)
tGPU = np.array(timeListGPU)
tCPU = np.array(timeListCPU)

# for i in range(len(nExamples)):
# Plotting both the curves simultaneously
plt.plot(n, tGPU, color='r', label='GPU')
plt.plot(n, tCPU, color='g', label='CPU')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Number of examples")
plt.grid()
plt.ylabel("Time,sec")
plt.title("GPU/CPU")
plt.ylim([-0.09, 0.3])

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
