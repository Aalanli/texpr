# %%
from typing import ClassVar, Dict, List, Optional, Callable, Sequence, Set, Tuple, Union
from enum import Enum
import gdsl.utils as utils


def valid_name(name: str) -> bool:
    if len(name) == 0:
        return False
    if name[0] == '%':
        return len(name) > 1 and ('x' + name[1:]).isidentifier()
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

    def __repr__(self) -> str:
        return str(self)


class DType(Type):
    def __init__(self, nbits: int, name: str):
        self.name: str = name
        self.nbits: int = nbits

    def __str__(self):
        return self.name

    def is_floating(self) -> bool:
        return self in (f64, f32, f16)

    def is_integral(self) -> bool:
        return self in (i64, i32, i16, i8, i1, u64, u32, u16, u8)


f64 = DType(64, 'f64')
f32 = DType(32, 'f32')
f16 = DType(16, 'f16')
i64 = DType(64, 'i64')
i32 = DType(32, 'i32')
i16 = DType(16, 'i16')
i8 = DType(8, 'i8')
i1 = DType(1, 'i1')
u64 = DType(64, 'u64')
u32 = DType(32, 'u32')
u16 = DType(16, 'u16')
u8 = DType(8, 'u8')


class TensorType(Type):
    def __init__(self, dtype: DType, shape: Tuple[Optional[int], ...]):
        self.dtype = dtype
        self.shape = shape

    def __str__(self):
        return f'{self.dtype}[{",".join(map(str, self.shape))}]'

    def __eq__(self, other):
        return isinstance(other, TensorType) and self.dtype == other.dtype and self.shape == other.shape

    def __hash__(self):
        return hash(str(self))


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
    def __init__(self, type: Type, name_hint: Optional[str] = None):
        assert isinstance(type, Type)
        self.type: Type = type
        self.name: str = '%' + str(id(self))
        self.name_hint = name_hint

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return isinstance(other, Value) and id(self) == id(other)

    def __str__(self) -> str:
        return self.name + ': ' + str(self.type)

    def __repr__(self) -> str:
        return str(self)


AttrTy = Union[int, float, bool, str, Tuple['AttrTy', ...], Sequence['AttrTy'], Dict[str, 'AttrTy']]


class Attr:
    def __init__(self, **attrs: AttrTy):
        assert len(attrs) > 0
        self.attrs = attrs

    def __eq__(self, other):
        return isinstance(other, Attr) and self.attrs == other.attrs

    def __hash__(self):
        return hash(str(self.attrs))


class Operation:
    def __init__(self, name: str, blocks: List['Block'], args: List['Value'], ret: List['Value']):
        self.name: str = name
        self.blocks: List['Block'] = blocks
        self.args: List['Value'] = args
        self.ret: List['Value'] = ret

    def lower_attr(self) -> Optional[Attr]:
        return None

    def __hash__(self):
        return id(self)

    def __str__(self):
        return IRPrinter().dump_op(self)

    def __repr__(self) -> str:
        return str(self)


class Block:
    def __init__(self, args: List['Value'], ops: List['Operation']):
        self.args: List['Value'] = args
        self.ops: List['Operation'] = ops

    def __hash__(self):
        return id(self)

    def __str__(self):
        return IRPrinter().dump_block(self)


# IR := op
# op := [(value | '(' value *(',' value) ')') '='] op_name arg_list [attr] [label] [block_list]
# op_name := IDENTIFIER
# attr := '{|' kv_value *(',' kv_value) '|}'
# kv_value := STRING ':' json
# json := INT | FLOAT | STRING | BOOL | dict | list
# dict := '{' kv_value *(',' kv_value) '}'
# list := '[' [json, *(',' json)] ']'
# arg_list := '(' ?(value) ')'
# block_list := '{' ?(block) '}'
# block := [label] '|' value *(',' value) '|' '{' *op '}'
# value := IDENTIFIER ':' TYPE
# label := (%NAT)


class Line:
    def __init__(self, s: str):
        assert '\n' not in s
        self.line = s

    def indent(self) -> 'Line':
        return Line('  ' + self.line)


class IRPrinter:
    def __init__(self, highlight_value: Optional[Set[Value]] = None, debug_labels: bool = False) -> None:
        self.highlight_value = set() if highlight_value is None else highlight_value
        self.debug_labels = debug_labels

        self.unique_names: Set[str] = set()

    def namer(self, x: Value) -> str:
        self.unique_names.add(x.name)
        return x.name

    def format_val_list(self, vals: List[Value], type_annot: bool = False) -> str:
        typer = lambda x: ': ' + str(x.type) if type_annot else ''
        namer = lambda x: self.namer(x) + typer(x)
        arg_it = map(lambda x: namer(x) if x not in self.highlight_value else utils.red(namer(x)), vals)
        return ', '.join(arg_it)

    def dump_attr_ty(self, attr: AttrTy) -> str:
        if isinstance(attr, (int, float, bool, str)):
            return str(attr)
        elif isinstance(attr, tuple):
            return '[' + ', '.join(map(self.dump_attr_ty, attr)) + ']'
        elif isinstance(attr, dict):
            kv = [k + ': ' + self.dump_attr_ty(v) for k, v in attr.items()]
            return '{' + ', '.join(kv) + '}'
        else:
            raise TypeError()

    def dump_attr(self, attr: Attr) -> str:
        return '{|' + ' ,'.join([k + ': ' + self.dump_attr_ty(v) for k, v in attr.attrs.items()]) + '|}'

    def dump_op_impl(self, ir: Operation) -> List[Line]:
        arg = self.format_val_list(ir.args)
        if len(ir.ret) > 0:
            ret = self.format_val_list(ir.ret, type_annot=True)
            if len(ir.ret) > 1:
                ret = '(' + ret + ')'
            header = Line(ret + ' = ' + ir.name + '(' + arg + ')')
        else:
            header = Line(ir.name + '(' + arg + ')')
        lines = [header]
        attr = ir.lower_attr()
        if attr is not None:
            line = ' ' + self.dump_attr(attr)
            lines[-1].line += line
        # the label
        if self.debug_labels:
            lines[-1].line += '%' + str(id(ir))
        if len(ir.blocks) > 0:
            if len(lines[-1].line) > 50:
                lines.append(Line('{'))
            else:
                lines[-1].line += ' {'
            for bk in ir.blocks:
                lines.extend([l.indent() for l in self.dump_block_impl(bk)])
            lines.append(Line('}'))
        return lines

    def dump_block_impl(self, ir: Block) -> List[Line]:
        label = ''
        if self.debug_labels:
            label = '%' + str(id(ir))
        header = Line(label + '|' + self.format_val_list(ir.args, type_annot=True) + '| {')
        lines = [header]
        for op in ir.ops:
            lines.extend([l.indent() for l in self.dump_op_impl(op)])
        lines.append(Line('}'))
        return lines

    def dump_op(self, ir: Operation) -> str:
        return '\n'.join(x.line for x in self.dump_op_impl(ir))

    def dump_block(self, block: Block) -> str:
        return '\n'.join(x.line for x in self.dump_block_impl(block))


class InvalidIRException(Exception):
    def __init__(self, ir: Operation, message: str, highlight_values: List[Value]) -> None:
        super().__init__(f"failed with {message} \n" + IRPrinter(highlight_value=set(highlight_values)).dump_op(ir))


def basic_verify_ir(ir: Operation):
    """verifies the basic SSA properties"""
    val_scopes: List[Set[Value]] = [set()]
    # every op/block object is used once
    op_block_codes = set()
    # every value has one definition/source
    val_sources = set()
    # every value has a unique name
    names = set()

    def is_defined(val: Value):
        for s in reversed(val_scopes):
            if val in s:
                return True
        return False

    def add_scope():
        val_scopes.append(set())

    def pop_scope():
        val_scopes.pop()

    def define_val(val: Value):
        val_scopes[-1].add(val)

    def verify_src_name(value: Value):
        name = value.name
        if not valid_name(name):
            raise InvalidIRException(ir, f"invalid name {name}", [value])
        if name in names:
            raise InvalidIRException(ir, f"name used more than once {name}", [value])
        names.add(name)

    def verify_block(b: Block):
        assert id(b) not in op_block_codes, "block object identify is used more than once"
        op_block_codes.add(id(b))

        for arg in b.args:
            assert not is_defined(arg), "repeated block arg"
            define_val(arg)
            verify_src_name(arg)
        for op in b.ops:
            verify_op(op)

    def verify_op(op: Operation):
        assert id(op) not in op_block_codes, "op object identify is used more than once"
        op_block_codes.add(id(op))
        for arg in op.args:
            if not is_defined(arg):
                print(IRPrinter(highlight_value={arg}, debug_labels=True).dump_op(ir))
                raise AssertionError("arg is not defined")
        for b in op.blocks:
            add_scope()
            verify_block(b)
            pop_scope()
        for r in op.ret:
            verify_src_name(r)
            assert not is_defined(r), "repeated op ret"
            assert r not in val_sources
            define_val(r)
            val_sources.add(r)

    try:
        verify_op(ir)
    except AssertionError as e:
        print("failed with", e)
        return False
    except InvalidIRException as e:
        print(e)
        return False
    return True
