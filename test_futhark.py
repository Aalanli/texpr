# %%
import pyopencl as cl
import numpy as np


a = np.random.rand(512, 512).astype(np.float32)
b = np.random.rand(512, 512).astype(np.float32)
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)



