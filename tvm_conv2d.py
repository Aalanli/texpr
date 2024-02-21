# %%
import numpy as np
import tvm
from tvm import te, auto_scheduler


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def conv2d(B, C1, H, W, C2, Kx, Ky, dilx, dily, strx, stry, dtype):
    I = te.placeholder((B, C1, H, W), name="I", dtype=dtype)
    F = te.placeholder((C2, C1, Ky, Kx), name="W", dtype=dtype)
    
    jc = te.reduce_axis((0, C1), name="jc")
    jx = te.reduce_axis((0, Kx), name="jx")
    jy = te.reduce_axis((0, Ky), name="jy")
    p = (H - dilx * (Kx - 1) - 1) // strx + 1
    q = (W - dily * (Ky - 1) - 1) // stry + 1
    out = te.compute(
        (B, C2, p, q),
        lambda ib, ic, ip, iq: te.sum(
            I[ib, jc, ip * strx + jx * dilx, iq * stry + jy * dily] * F[ic, jc, jy, jx],
            axis=[jc, jx, jy],
        )
    )

    return [I, F, out]


target = tvm.target.Target("cuda")
B = 1
C1 = 128
C2 = 256
H = 56
W = 56
Kx = 3
Ky = 3
dilx = 1
dily = 1
strx = 1
stry = 1
args = (B, C1, H, W, C2, Kx, Ky, dilx, dily, strx, stry, "float32")

task = auto_scheduler.SearchTask(func=conv2d, args=args, target=target)

# Inspect the computational graph
# # print("Computational DAG:")
print(task.compute_dag)
# 
log_file = "conv2d3x3.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
# task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

import torch

func = tvm.build(sch, args, target)
I_tch = torch.randn([B, C1, H, W], dtype=torch.float32, device="cuda")
W_tch = torch.randn([C2, C1, Ky, Kx], dtype=torch.float32, device="cuda")
 
out_tch = torch.nn.functional.conv2d(I_tch, W_tch, stride=(strx, stry), dilation=(dilx, dily))
 
dev = tvm.cuda()
i_tvm = tvm.nd.array(I_tch.cpu().numpy(), device=dev)
w_tvm = tvm.nd.array(W_tch.cpu().numpy(), device=dev)
out_tvm = tvm.nd.empty(out_tch.shape, device=dev)
func(i_tvm, w_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_tch.cpu().numpy(), out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(i_tvm, w_tvm, out_tvm).results) * 1000)
)

import triton
import torch

print(triton.testing.do_bench(lambda: func(i_tvm, w_tvm, out_tvm)))
print(triton.testing.do_bench(lambda: torch.nn.functional.conv2d(I_tch, W_tch, stride=(strx, stry), dilation=(dilx, dily))))
