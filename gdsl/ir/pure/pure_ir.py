# %%
from typing import ClassVar, Dict, List, Optional, Callable, Set, Tuple, Union
from enum import Enum
import gdsl.utils as utils

from ..base_ir import *

class BinaryOpCodes(Enum):
    Add = 0
    Mul = 1
    Div = 2
    Mod = 3

    And = 4
    Or = 5

    Eq = 6
    Lt = 7

    LAnd = 8
    LOr = 9

    def type_check(self, lhs: DType, rhs: DType) -> bool:
        assert isinstance(lhs, DType) and isinstance(rhs, DType)
        if lhs != rhs:
            return False
        if self.value in (0, 1, 2, 6, 7):
            return lhs != i1
        if self.value in (3, 4, 5):
            return lhs.is_integral()
        if self.value in (8, 9):
            return lhs == i1
        return True

    def ret_ty(self, lhs: DType, rhs: DType) -> DType:
        assert self.type_check(lhs, rhs)
        if self.value in (0, 1, 3, 4, 5, 8, 9):
            return lhs
        if self.value == 2:
            if lhs.is_floating() or rhs.is_floating():
                return lhs if lhs.is_floating() else rhs
            else:
                return lhs
        if self.value in (6, 7):
            return i1
        assert False


class ReduceOperations(Enum):
    Sum = 0
    Min = 1
    Max = 2


class ConstantOp(Operation):
    def __init__(self, py_constant):
        assert isinstance(py_constant, (int, bool, float))
        self.py_constant = py_constant
        if isinstance(py_constant, int):
            ty = i32
        elif isinstance(py_constant, bool):
            ty = i1
        elif isinstance(py_constant, float):
            ty = f32
        super().__init__('constant', [], [], [Value(ty)])

    def lower_attr(self) -> Optional[Attr]:
        return Attr(val=self.py_constant)


class YieldOp(Operation):
    def __init__(self, val: List[Value]):
        super().__init__('yield', [], val, [])


class BinaryOp(Operation):
    def __init__(self, op: BinaryOpCodes, lhs: Value, rhs: Value, ret: Value):
        super().__init__(op.name.lower(), [], [lhs, rhs], [ret])


class ElementwiseMathOp(Operation):
    def __init__(self, op_code: str, x: Value):
        assert isinstance(x.type, DType) and x.type.is_floating()
        self.op_code = op_code
        ret = Value(x.type)
        super().__init__('elementwise', [], [x], [ret])

    def lower_attr(self) -> Optional[Attr]:
        return Attr(op_code=self.op_code)


class NotOp(Operation):
    def __init__(self, value: Value, ret: Value):
        super().__init__('not', [], [value], [ret])


class NegOp(Operation):
    def __init__(self, value: Value, ret: Value):
        super().__init__('neg', [], [value], [ret])


class IfThenElseOp(Operation):
    def __init__(self, cond: Value, true_value: Value, false_value: Value, ret: Value):
        super().__init__('if', [], [cond, true_value, false_value], [ret])


class GridComputeOp(Operation):
    def __init__(self, shape: Tuple['Value', ...], block: Block, ret: Value):
        super().__init__('grid_compute', [block], list(shape), [ret])


class GridReduceOp(Operation):
    def __init__(self, reduction: ReduceOperations, shape: Tuple['Value', ...], block: Block, ret: Value):
        super().__init__('grid_reduce', [block], list(shape), [ret])
        self.reduction = reduction

    def lower_attr(self) -> Optional[Attr]:
        return Attr(reduction=self.reduction.name)


class TensorIndexOp(Operation):
    def __init__(self, tensor: Value, indices: Tuple['Value', ...], ret: Value):
        super().__init__('index', [], [tensor, *indices], [ret])


class TensorShapeOfOp(Operation):
    def __init__(self, tensor: Value, index: int):
        assert isinstance(tensor.type, TensorType)
        val = Value(i32, 's')
        self.index = index
        super().__init__('shape_of', [], [tensor], [val])

    def lower_attr(self) -> Optional[Attr]:
        return Attr(index=self.index)

    def tied_to(self) -> Optional[Value]:
        assert isinstance(self.args[0].type, TensorType)
        v = self.args[0].type.shape[self.index]
        return v if isinstance(v, Value) else None


class AssertOp(Operation):
    def __init__(self, cond: Value):
        assert cond.type == i1
        super().__init__('static_assert', [], [cond], [])


class IRFunctionOp(Operation):
    def __init__(self, body: Block, name: str):
        assert isinstance(body.ops[-1], YieldOp)
        fn_ty = FunctionType([a.type for a in body.args], [body.ops[-1].args[0].type])
        ret_var = Value(fn_ty, name_hint=name)
        self.fn_name = name
        super().__init__('function', [body], [], [ret_var])
    
    def lower_attr(self) -> Optional[Attr]:
        return Attr(name=self.fn_name)


class IRModuleOp(Operation):
    def __init__(self, scope: Block):
        super().__init__('module', [scope], [], [])


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
