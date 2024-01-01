# %%
from typing import Tuple
from gdsl import *
from gdsl.ir.pure import Block


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
        for block in op.blocks:
            self.run_on_block(block)


class OpValueDependenceAnalysis(Pass):
    def __init__(self) -> None:
        self.val_source: Dict[Value, Union[Operation, Block]] = {}
        self.val_users: Dict[Value, Set[Operation]] = {}

    def add_use(self, val: Value, op: Operation):
        if val not in self.val_users:
            self.val_users[val] = set()
        self.val_users[val].add(op)

    def run_on_op(self, op: Operation):
        for oper in op.args:
            self.add_use(oper, op)
        for ret in op.ret:
            self.val_source[ret] = op
        return super().run_on_op(op)
    
    def run_on_block(self, block: Block):
        for arg in block.args:
            self.val_source[arg] = block
        return super().run_on_block(block)


class PureLICM(Pass):
    def run_on_op(self, op: Operation):
        super().run_on_op(op)
        for block in op.blocks:
            i = 0
            while i < len(block.ops):
                l_op = block.ops[i]
                if isinstance(l_op, (GridComputeOp, GridReduceOp)):
                    assert len(l_op.blocks) == 1
                    licm = self.on_block(l_op.blocks[0])
                    if len(licm) > 0:
                        block.ops = block.ops[:i] + licm + [l_op] + block.ops[i + 1 :]
                        i += len(licm) + 1
                    else:
                        i += 1
                else:
                    i += 1

    def on_block(self, block: Block) -> List[Operation]:
        worklist: Set[Value] = set(block.args)
        touched_ops = set()
        while len(worklist) > 0:
            new_worklist = set()
            for op in block.ops:
                if op in touched_ops:
                    continue
                for oper in op.args:
                    if oper in worklist:
                        touched_ops.add(op)
                        for ret in op.ret:
                            new_worklist.add(ret)
                        break
            worklist = new_worklist

        new_block_ops = []
        removed = []
        for op in block.ops:
            if op in touched_ops:
                new_block_ops.append(op)
            else:
                removed.append(op)
        block.ops = new_block_ops
        if len(removed) > 0 and isinstance(removed[-1], YieldOp):
            removed.pop()
        return removed


class CollectOp(Pass):
    def __init__(self) -> None:
        self.ops: List[Operation] = []

    def run_on_op(self, op: Operation):
        self.ops.append(op)
        super().run_on_op(op)


class CollectOpReturn(Pass):
    def __init__(self, level: Optional[int] = None) -> None:
        self.values: List[Value] = []
        self.level = level
        self.cur_level = 0

    def run_on_op(self, op: Operation):
        if self.level is not None and self.cur_level > self.level:
            return
        self.cur_level += 1
        super().run_on_op(op)
        self.values.extend(op.ret)
        self.cur_level -= 1


class CollectBlockArgs(Pass):
    def __init__(self) -> None:
        self.values: List[Value] = []

    def run_on_block(self, block: Block):
        self.values.extend(block.args)
        super().run_on_block(block)


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
        prefix = op.name[0] if len(op.name) > 0 else ''
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


class CollectPureOpImplicitReference(Pass):
    def __init__(self) -> None:
        self.implicit_uses: Dict[Operation, Set[Value]] = {}
        self.implicit_users: Dict[Value, Set[Operation]] = {}

    def on_op(self, op: Operation) -> Set[Value]:
        vals: Set[Value] = set()
        for block in op.blocks:
            vals = vals.union(self.on_block(block))
        vals.update(op.args)
        self.implicit_uses[op] = vals
        return vals

    def on_block(self, block: Block) -> Set[Value]:
        vals: Set[Value] = set()
        diff: Set[Value] = set()
        for op in block.ops:
            vals = vals.union(self.on_op(op))
            diff.update(op.ret)
        for arg in block.args:
            vals.remove(arg)
        return vals.difference(diff)

    def run_on_op(self, op: Operation):
        self.on_op(op)
        for op, vals in self.implicit_uses.items():
            for v in vals:
                if v not in self.implicit_users:
                    self.implicit_users[v] = set()
                self.implicit_users[v].add(op)


def replace_operand(op: Operation, new: Value, old: Value):
    for i, oper in enumerate(op.args):
        if oper == old:
            op.args[i] = new


class PureGreedyCSE(Pass):
    def __init__(self, capture: CollectPureOpImplicitReference, dep: OpValueDependenceAnalysis):
        self.capture = capture
        self.dep = dep

    @staticmethod
    def equivalent(op1: Operation, op2: Operation):
        if type(op1) != type(op2):
            return False
        if op1.lower_attr() != op2.lower_attr():
            return False
        if len(op1.ret) != len(op2.ret):
            return False
        return True

    def run_on_block(self, block: Block):
        super().run_on_block(block)

        removed: Set[Value] = set()

        collect = CollectOpReturn(level=0)
        collect.run_on_block(block)

        collect2 = CollectOpReturn()
        collect2.run_on_block(block)

        # vals are all unique if ir is correct
        vals: List[Value] = collect.values
        vals2 = collect.values
        for i in range(len(vals)):
            if vals[i] in removed:
                continue
            for j in range(len(vals2)):
                if vals[j] in removed:
                    continue
                op1 = self.dep.val_source[vals[i]]
                op2 = self.dep.val_source[vals[j]]

                if op1 is op2 or isinstance(op1, Block) or isinstance(op2, Block):  # come from the same op
                    continue
                if not PureGreedyCSE.equivalent(op1, op2):
                    continue
                if isinstance(op1, YieldOp):
                    continue
                # print("op1", self.capture.implicit_uses[op1])
                # print(op1)

                # print("op2", self.capture.implicit_uses[op2])
                # print(op2)

                if self.capture.implicit_uses[op1] != self.capture.implicit_uses[op2]:
                    continue
                # print(op1.name)

                # replace all returns of op2 with returns of op1
                for r1, r2 in zip(op1.ret, op2.ret):
                    for explict_user in self.dep.val_users[r2]:
                        replace_operand(explict_user, r1, r2)
                    for implicit_user in self.capture.implicit_users[r2]:
                        self.capture.implicit_uses[implicit_user].remove(r2)
                        self.capture.implicit_uses[implicit_user].add(r1)
                    removed.add(r2)


class PureLivelinessAnalysis(Pass):
    def __init__(self) -> None:
        self.alive: Set[Value] = set()

    def run_on_op(self, op: Operation):
        if isinstance(op, IRFunctionOp):
            self.alive.add(op.ret[0])
        if any(r in self.alive for r in op.ret):
            for block in op.blocks:
                assert isinstance(block.ops[-1], YieldOp)
                for arg in block.ops[-1].args:
                    self.alive.add(arg)
        return super().run_on_op(op)

    def run_on_block(self, block: Block):
        for op in reversed(block.ops):
            if any(r in self.alive for r in op.ret):
                self.alive.update(op.args)
                self.run_on_op(op)


class PureDce(Pass):
    def __init__(self, liveliness: PureLivelinessAnalysis):
        self.alive = liveliness

    def run_on_block(self, block: Block):
        new_ops = []
        for op in block.ops:
            if isinstance(op, YieldOp) or any(r in self.alive.alive for r in op.ret):
                new_ops.append(op)
        block.ops = new_ops
        return super().run_on_block(block)


class VecGridComputeOp(Operation):
    def __init__(self, shape: Tuple[Value, ...], tile: Tuple[Value,...], block: Block, ret: Value):
        assert all(s.type == i32 for s in shape)
        assert all(t.type == i32 for t in tile)
        super().__init__('grid_compute_tile', [block], shape + tile, [ret])
        self.shape_len = len(shape)

PrettyRenameValues().run_on_op(ir_module)
print(ir_module)
print(ir.basic_verify_ir(ir_module))



