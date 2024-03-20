# %%
import copy
from typing import ClassVar, Dict, List, Optional, Callable, Set, Tuple, Union, cast
from enum import Enum
from gdsl.ir.base_ir import Attr, Block, List, Operation, Type
import gdsl.utils as utils
from mypy_extensions import trait

from ..base_ir import *


@trait
class ElementWiseOpCode:
    def type_check(self, types: List[DType]) -> bool:
        assert False

    def result_dtype(self, types: List[DType]) -> DType:
        assert False

    def name_of(self) -> str:
        assert False

Dim = List[Optional[int]]
def broadcast_many(shapes: List[Dim]) -> Dim:
    shape = shapes[0]
    for i in range(1, len(shapes)):
        shape = broadcast(shape, shapes[i])
    return shape

def broadcast(sh1: Dim, sh2: Dim) -> Dim:
    if len(sh1) < len(sh2):
        return broadcast(sh2, sh1)
    sh2_ = [1] * (len(sh1) - len(sh2)) + sh2
    new_shape = []
    for i, j in zip(sh1, sh2_):
        if i == 1:
            new_shape.append(j)
        elif j == 1:
            new_shape.append(i)
        elif i == j:
            new_shape.append(i)
        else:
            new_shape.append(None)
    return new_shape

class BinaryOpCodes(ElementWiseOpCode, Enum):
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

    def type_check(self, types: List[DType]) -> bool:
        assert len(types) == 2
        lhs, rhs = types
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

    def result_dtype(self, types: List[DType]) -> DType:
        self.type_check(types)
        lhs, rhs = types
        assert isinstance(lhs, DType) and isinstance(rhs, DType)
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

    def name_of(self) -> str:
        return self.name.lower()


class UnaryOpCodes(ElementWiseOpCode, Enum):
    Neg = 0
    Not = 1

    def type_check(self, types: List[DType]) -> bool:
        assert len(types) == 1
        ty = types[0]
        assert isinstance(ty, DType)

        if self.value == 0:
            return ty.is_floating() or ty.is_integral()
        else:
            return ty.is_integral() or ty == i1

    def result_dtype(self, types: List[DType]) -> DType:
        assert self.type_check(types)
        return types[0]

    def name_of(self) -> str:
        return self.name.lower()


class FloatIntrinsicOpCode(ElementWiseOpCode):
    def __init__(self, op_code: str):
        self.op_code = op_code

    def type_check(self, types: List[DType]) -> bool:
        if len(types) == 0:
            return False
        t1 = types[0]
        if not (isinstance(t1, DType) and t1.is_floating()):
            return False
        return all(t == t1 for t in types)

    def result_dtype(self, types: List[DType]) -> DType:
        self.type_check(types)
        return types[0]

    def name_of(self) -> str:
        return self.op_code


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

    def attrs(self) -> Optional[Attr]:
        return Attr(val=self.py_constant)


class YieldOp(Operation):
    def __init__(self, val: List[Value]):
        super().__init__('yield', [], val, [])


class ElementwiseOp(Operation):
    def __init__(self, op: ElementWiseOpCode, args: Tuple[Value, ...]):
        ret = Value(ElementwiseOp.get_res_type(op, [a.type for a in args]))
        super().__init__('elementwise', [], list(args), [ret])
        self.elem_op = op

    def attrs(self) -> Optional[Attr]:
        return Attr(op=self.elem_op.name_of())
    
    @staticmethod
    def get_res_type(op: ElementWiseOpCode, types: List[Type]) -> Type:
        if all(isinstance(t, DType) for t in types):
            return op.result_dtype(cast(List[DType], types))
        assert all(isinstance(t, DType) or isinstance(t, TensorType) for t in types)
        dtypes_ = [t.dtype if isinstance(t, TensorType) else t for t in types]
        dtypes = cast(List[DType], dtypes_)
        res_ty = op.result_dtype(dtypes)
        shapes = [list(t.shape) if isinstance(t, TensorType) else [1] for t in types]
        
        return TensorType(res_ty, tuple(broadcast_many(shapes)))


class ExpandDimOp(Operation):
    def __init__(self, tensor: Value, idx: int):
        assert isinstance(tensor.type, TensorType)
        assert 0 <= idx <= len(tensor.type.shape)
        res_shape = tensor.type.shape[:idx] + (1,) + tensor.type.shape[idx:]
        res = Value(TensorType(tensor.type.dtype, res_shape))
        super().__init__("expand_dim", [], [tensor], [res])


class GridComputeOp(Operation):
    def __init__(self, shape: Tuple['Value', ...], block: Block, ret: Value):
        super().__init__('grid_compute', [block], list(shape), [ret])


class GridReduceOp(Operation):
    def __init__(self, reduction: ReduceOperations, shape: Tuple['Value', ...], block: Block, ret: Value):
        super().__init__('grid_reduce', [block], list(shape), [ret])
        self.reduction = reduction

    def attrs(self) -> Optional[Attr]:
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

    def attrs(self) -> Optional[Attr]:
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

    def attrs(self) -> Optional[Attr]:
        return Attr(name=self.fn_name)
    

class IRModuleOp(Operation):
    def __init__(self, scope: Block):
        super().__init__('module', [scope], [], [])
