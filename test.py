# %%
import copy
from typing import Optional, Tuple
from gdsl import *
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


with Trace() as builder:
    builder.build_fn(
        "matmul", [tensor((32, 32), ir.f32), tensor((32, 32), ir.f32)], matmul
    )
    builder.build_fn("softmax", [tensor((None,), ir.f32)], softmax)
    builder.build_fn('conv2d', [tensor((4, 256, 128, 128), ir.f32), tensor((256, 256, 3, 3), ir.f32)], lambda im, w: conv2d(im, w, stride=(1, 1), dilation=(1, 1)))

ir_module = builder.finish_ir_module()

from gdsl.ir.pure import *


class Pass:
    def run_on_block(self, block: Block):
        for op in block.ops:
            self.run_on_op(op)

    def run_on_op(self, op: Operation):
        for block in op.blocks():
            self.run_on_block(block)


def fill_with_name(i: int) -> str:
    ia = ord('a')
    # 0->a, 25->z
    # 26->aa,
    n = ''
    while True:
        n += chr(ia + i % 26)
        i = i // 26
        if i <= 0:
            break
        i -= 1
    return n[::-1]


class PrettyRenameValues(Pass):
    def __init__(self) -> None:
        self.prefix_bases: Dict[str, int] = {'': 0}
        self.unique_names: Set[str] = {''}
        self.uid: int = 0

    def generate_name(self, prefix: Optional[str] = None) -> str:
        if prefix is None:
            prefix = ''
        if prefix not in self.prefix_bases:
            self.prefix_bases[prefix] = 0
            return prefix

        max_base_len = 26**3 + 26**2 + 26
        base = self.prefix_bases[prefix]
        name = prefix + fill_with_name(base % max_base_len)
        if base >= max_base_len:
            name += str(base - max_base_len)
        self.prefix_bases[prefix] += 1
        return name

    def generate_unique_name(self, prefix: Optional[str] = None) -> str:
        name = self.generate_name(prefix)
        while name in self.unique_names:
            name = self.generate_name(prefix)
        return name
    
    def generate_unique_block_name(self) -> str:
        name = '%' + str(self.uid)
        assert name not in self.unique_names
        self.uid += 1
        self.unique_names.add(name)
        return name

    def run_on_op(self, op: Operation):
        prefix = op.name()[0] if len(op.name()) > 0 else ''
        for r in op.ret:
            if r.name_hint is None:
                r.name = self.generate_unique_name(prefix)
            else:
                r.name = self.generate_unique_name(r.name_hint)
        super().run_on_op(op)
    
    def run_on_block(self, block: Block):
        for a in block.args:
            a.name = self.generate_unique_block_name()
        return super().run_on_block(block)

def renamer_on_op(namer: PrettyRenameValues, op: Op) -> Op:
    prefix = op.name()[0] if len(op.name()) > 0 else ''

    if isinstance(op, Operation):
        op = copy.copy(op)
        for r in op.ret:
            if r.name_hint is None:
                r.name = namer.generate_unique_name(prefix)
            else:
                r.name = namer.generate_unique_name(r.name_hint)
        
    

def pretty_rename_values(op: Op) -> Op:
    namer = PrettyRenameValues()

        
