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

    def type_check(self, lhs: Type, rhs: Type) -> bool:
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
