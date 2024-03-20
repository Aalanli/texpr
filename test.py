# %%
from dataclasses import dataclass
import copy
from typing import Optional, Tuple, List, Dict, Set
from gdsl.lang import *
from gdsl.ir.base_ir import Attr
from gdsl.ir.pure import Block, Operation


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


def conv2d(im: Var, weight: Var, stride: Tuple[int, int], dilation: Tuple[int, int]) -> Var:
    assert len(tensor_shape(im)) == len(tensor_shape(weight)) == 4
    b, c1, h, w = [shape_of(im, i) for i in range(4)]
    c2, c11, kx, ky = [shape_of(weight, i) for i in range(4)]
    sassert(c11 == c1)
    dilx, dily = dilation
    stx, sty = stride

    p = (h - (kx - 1) * dilx - 1) // stx + 1
    q = (w - (ky - 1) * dily - 1) // sty + 1
    y = grid_compute(
        (b,c2,p,q),
        f=lambda bi, c2i, pi, qi: grid_reduce(
            reduction=ir.ReduceOperations.Sum,
            shape=(c1, kx, ky),
            f=lambda jc, jx, jy: im[bi, jc, pi * stx + jx * dilx, qi * sty + jy * dily] * \
                weight[c2i, jc, jx, jy]
        ))
    return y


from gdsl.ir import Op, Value, Operation

def predef_ops(op: Op) -> List[Op]:
    ops = []
    worklist = [op]
    while len(worklist) > 0:
        op = worklist.pop()
        ops.append(op)
        for block in op.blocks():
            for op in block.ops:
                worklist.append(op)
    return ops


def collect_op_returns(op: Op) -> List[Value]:
    values = []
    for op in predef_ops(op):
        values.extend(op.returns())
    return values


def collect_block_args(op: Op) -> List[Value]:
    values = []
    for op in predef_ops(op):
        for block in op.blocks():
            values.extend(block.args)
    return values

class KernelGrid(Operation):
    def __init__(self, grid_dim: Value, block_dim: Value, ret: Value):
        grid_idx = Value(ir.i32, 'grid_idx_x')
        block_idx = Value(ir.i32, 'block_idx_x')
        block = Block([grid_idx, block_idx], [])
        super().__init__('kernel_grid', [block], [grid_dim, block_dim], [ret])


class ForLoop(Operation):
    def __init__(self, count: Value, block: Block, args: List[Value]):
        super().__init__('for', [block], [count] + args, [])

    def carried_vars(self) -> List[Value]:
        return self.args[1:]


def canoncanize_compute(op: Op) -> Op:
    pass


def canoncanize_reduce(op: Op) -> Op:
    pass


def lower_kernel_grid(op: Op) -> Op:
    # TODO: lower compute and reduce to kernel and for loop
    pass


with Trace() as builder:
    builder.build_fn(
        "matmul", [tensor((32, 32), ir.f32), tensor((32, 32), ir.f32)], matmul
    )
    builder.build_fn("softmax", [tensor((None,), ir.f32)], softmax)
    builder.build_fn('conv2d', [tensor((4, 256, 128, 128), ir.f32), tensor((256, 256, 3, 3), ir.f32)], lambda im, w: conv2d(im, w, stride=(1, 1), dilation=(1, 1)))

ir_module = builder.finish_ir_module()
print(ir_module)

