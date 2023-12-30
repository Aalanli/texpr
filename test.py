# %%
from gdsl import *


def matmul(a: Var, b: Var):
    assert len(tensor_shape(a)) == len(tensor_shape(b)) == 2
    m, k = shape_of(a, 0), shape_of(a, 1)
    k1, n = shape_of(b, 0), shape_of(b, 1)
    sassert(k == k1)

    c = grid_compute(
        (m, n),
        f=lambda i, j: grid_reduce(
            ir.ReduceOperations.Sum, (k,), f=lambda k: a[i, k] * b[k, j]
        ),
    )
    return c


def softmax(a: Var):
    assert len(tensor_shape(a)) == 1
    n = shape_of(a, 0)
    maxes = grid_reduce(ir.ReduceOperations.Max, (n,), lambda i: a[i])
    subs = grid_compute((n,), lambda i: exp(a[i] - maxes))
    sums = grid_reduce(ir.ReduceOperations.Sum, (n,), lambda i: subs[i])
    y = grid_compute((n,), lambda i: subs[i] / sums)
    return y


with Trace() as builder:
    builder.build_fn(
        "matmul", [tensor((32, 32), ir.f32), tensor((32, 32), ir.f32)], matmul
    )
    builder.build_fn("softmax", [tensor((None,), ir.f32)], softmax)

ir_module = builder.ir_module()

print(ir_module)
