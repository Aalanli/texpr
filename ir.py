# %%
from typing import List, Optional, Callable, Tuple, Union
from enum import Enum

def valid_name(name: str) -> bool:
    return name.isidentifier()

def is_base_value(value):
    if isinstance(value, (int, float, bool, str)):
        return True
    if isinstance(value, tuple):
        return all(map(is_base_value, value))
    return False

class Type:
    """
    Base class for all types.
    There exists only one instance for each type.
    Thus we can use `is` to compare two types.
    """
    def __hash__(self):
        return id(self)

    def __eq__(self, other: 'Type'):
        return self is other
    
    def __str__(self):
        raise NotImplementedError()

class DType(Type):
    def __init__(self, nbits: int, name: str):
        self.name: str = name
        self.nbits: int = nbits
    
    def __str__(self):
        return self.name

f64 = DType(64, 'f64')
f32 = DType(32, 'f32')
f16 = DType(16, 'f16')
i64 = DType(64, 'i64')
i32 = DType(32, 'i32')
i16 = DType(16, 'i16')
i8  = DType(8, 'i8')
i1  = DType(1, 'i1')
u64 = DType(64, 'u64')
u32 = DType(32, 'u32')
u16 = DType(16, 'u16')
u8  = DType(8, 'u8')


class TensorType(Type):
    def __init__(self, dtype: DType, shape: Tuple[Union[int, 'Value']]):
        self.dtype: DType = dtype
        self.shape: Tuple['Value'] = shape
    
    def __str__(self):
        return f'{self.dtype}[{",".join(map(str, self.shape))}]'

class FunctionType(Type):
    def __init__(self, args: List[Type], ret: List[Type]):
        self.args = args
        self.ret = ret
    
    def __str__(self):
        header = '(' + ', '.join(map(str, self.args)) + ') -> '
        if len(self.ret) == 0:
            header += '()'
        elif len(self.ret) == 1:
            header += str(self.ret[0])
        else:
            header += '(' + ', '.join(map(str, self.ret)) + ')'
        return header
    
    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.args == other.args and self.ret == other.ret
    
    def __hash__(self):
        return hash(('fn_ty', tuple(self.args), tuple(self.ret)))
 

class Value:
    _uid: int = 0
    def __init__(self, type: Type, name: Optional[str] = None):
        assert isinstance(type, Type)
        self.type: Type = type
        self.uid: int = Value._uid
        self.name: Optional[str] = name
        if name is not None:
            assert len(name) > 0
            if name[0] != '%':
                assert valid_name(name), f'invalid name: {name}'

        Value._uid += 1
    
    def __hash__(self):
        return hash(self.uid)
    
    def __eq__(self, other: 'Value'):
        return self.uid == other.uid
    
    def __str__(self):
        if self.name is None:
            return f'%{self.uid}'
        else:
            return f'%{self.name}{self.uid}'


class Operation:
    def __init__(self, name: str, blocks: list['Block'], args: list['Value'], ret: list['Value']):
        self.name: str = name
        self.blocks: list['Block'] = blocks
        self.args: list['Value'] = args
        self.ret: list['Value'] = ret


class Block:
    def __init__(self, args: list['Value'], ops: list['Operation']):
        self.args: list['Value'] = args
        self.ops: list['Operation'] = ops

    

class BinaryOpCodes(Enum):
    Add = 0
    Mul = 1
    Div = 2
    Mod = 3

    And = 4
    Or  = 5

    Eq = 6
    Lt = 7

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

class YieldOp(Operation):
    def __init__(self, val: Value):
        super().__init__('yield', [], [val], [])

class BinaryOp(Operation):
    def __init__(self, op: BinaryOpCodes, lhs: Value, rhs: Value, ret: Value):
        super().__init__(op.name.lower(), [], [lhs, rhs], [ret])

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
    def __init__(self, shape: Tuple['Value'], block: Block, ret: Value):
        super().__init__('grid_compute', [block], shape, [ret])

class GridReduceOp(Operation):
    def __init__(self, reduction: ReduceOperations, shape: Tuple['Value'], block: Block, ret: Value):
        super().__init__('grid_reduce', [block], shape, [ret])

class TensorIndexOp(Operation):
    def __init__(self, tensor: Value, indices: Tuple['Value'], ret: Value):
        super().__init__('index', [], [tensor, *indices], [ret])

class IRFunctionOp(Operation):
    def __init__(self, body: Block, name: str):
        assert isinstance(body.ops[-1], YieldOp)
        fn_ty = FunctionType([a.type for a in body.args], [body.ops[-1].args[0].type])
        ret_var = Value(fn_ty, name=name)
        self.fn_name = name
        super().__init__('function', [body], [], [ret_var])

class IRModuleOp(Operation):
    def __init__(self, functions: List[IRFunctionOp]):
        super().__init__('module', [], [], [])
        self.ir_functions = functions

