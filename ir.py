# %%
from typing import Dict, List, Optional, Callable, Set, Tuple, Union
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

    def __eq__(self, other):
        return isinstance(other, Type) and self is other
    
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
    def __init__(self, dtype: DType, shape: Tuple[Union['Value', int], ...]):
        self.dtype: DType = dtype
        self.shape: Tuple[Union['Value', int], ...] = shape
    
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
    
    def __eq__(self, other):
        return isinstance(other, Value) and self.uid == other.uid
    
    def __str__(self):
        if self.name is None:
            return f'%{self.uid}'
        else:
            return f'%{self.name}{self.uid}'


AttrTy = Union[int, float, bool, str, Tuple['AttrTy', ...], Dict[str, 'AttrTy']]        

class Attr:
    def __init__(self, **attrs: Dict[str, AttrTy]):
        assert len(attrs) > 0
        self.attrs = attrs
    
class Operation:
    def __init__(self, name: str, blocks: list['Block'], args: list['Value'], ret: list['Value']):
        self.name: str = name
        self.blocks: list['Block'] = blocks
        self.args: list['Value'] = args
        self.ret: list['Value'] = ret
    
    def lower_attr(self) -> Optional[Attr]:
        return None


class Block:
    def __init__(self, args: list['Value'], ops: list['Operation'], label: Optional[str] = None):
        self.args: list['Value'] = args
        self.ops: list['Operation'] = ops
        self.label = label
        if label is not None:
            assert label.isidentifier()

    

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
    def lower_attr(self) -> Attr | None:
        return Attr(val=self.py_constant)

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
    def __init__(self, shape: Tuple['Value',...], block: Block, ret: Value):
        super().__init__('grid_compute', [block], list(shape), [ret])

class GridReduceOp(Operation):
    def __init__(self, reduction: ReduceOperations, shape: Tuple['Value',...], block: Block, ret: Value):
        super().__init__('grid_reduce', [block], list(shape), [ret])

class TensorIndexOp(Operation):
    def __init__(self, tensor: Value, indices: Tuple['Value', ...], ret: Value):
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


# IR := op
# op := [(value | '(' value *(',' value) ')') '='] op_name arg_list [attr] [block_list]
# op_name := IDENTIFIER
# attr := '{|' kv_value *(',' kv_value) '|}'
# kv_value := STRING ':' json
# json := INT | FLOAT | STRING | BOOL | dict | list
# dict := '{' kv_value *(',' kv_value) '}'
# list := '[' [json, *(',' json)] ']'
# arg_list := '(' ?(value) ')'
# block_list := '{' ?(block) '}'
# block := [label] '|' value *(',' value) '|' '{' *op '}'
# label := IDENTIFIER
# value := '%' IDENTIFIER ':' TYPE

class Line:
    def __init__(self, s: str):
        assert '\n' not in s
        self.line = s
    
    def indent(self) -> 'Line':
        return Line('  ' + self.line)

class ValueNamer:
    def __init__(self) -> None:
        self.names: Dict[Value, str] = {}
        self.counter = 0
        self.used_names: Set[str] = set()
    
    def name(self, v: Value) -> str:
        if v in self.names:
            return self.names[v]
        else:
            if v.name is not None:
                name = v.name
                idx = 0
                while name + str(idx) in self.used_names:
                    idx += 1
                name = '%' + name + str(idx)
            else:
                name = f'%{self.counter}'
            name = name + ': ' + str(v.type)
            self.counter += 1
            self.names[v] = name
            return name

def dump_ir(ir: Operation):
    namer = ValueNamer()

    def dump_attr_ty(attr: AttrTy) -> str:
        if isinstance(attr, (int, float, bool, str)):
            return str(attr)
        elif isinstance(attr, tuple):
            return '[' + ', '.join(map(dump_attr_ty, attr)) + ']'
        elif isinstance(attr, dict):
            kv = [k + ': ' + dump_attr_ty(v) for k, v in attr.items()]
            return '{' + ', '.join(kv) + '}'
        else:
            raise TypeError()
    
    def dump_attr(attr: Attr) -> str:
        return '{|' + ' ,'.join([k + ': ' + dump_attr_ty(v) for k, v in attr.attrs.items()]) + '|}'
    
    def dump_op(ir: Operation) -> List[Line]:
        arg = ', '.join(map(namer.name, ir.args))
        if len(ir.ret) > 0:
            ret = ', '.join(map(namer.name, ir.ret))
            if len(ir.ret) > 1:
                ret = '(' + ret + ')'
            header = Line(ret + ' = ' + ir.name + '(' + arg + ')')
        else:
            header = Line(ir.name + '(' + arg + ')')
        lines = [header]
        attr = ir.lower_attr()
        if attr is not None:
            line = ' ' + dump_attr(attr)
            lines[-1].line += line
        if len(ir.blocks) > 0:
            if len(lines[-1].line) > 50:
                lines.append(Line('{'))
            else:
                lines[-1].line += ' {'
            for bk in ir.blocks:
                lines.extend([l.indent() for l in dump_block(bk)])
            lines.append(Line('}'))
        return lines

    def dump_block(ir: Block) -> List[Line]:
        label = '' if ir.label is None else ir.label + ' '
        header = Line(label + '|' + ', '.join(map(namer.name, ir.args)) + '| {')
        lines = [header]
        for op in ir.ops:
            lines.extend([l.indent() for l in dump_op(op)])
        lines.append(Line('}'))
        return lines

    return '\n'.join([l.line for l in dump_op(ir)])
