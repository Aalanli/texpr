# %%
from typing import Callable, ClassVar, List, Tuple, Union, Set, Optional
from .ir.pure import ir
from .ir.pure.ir import Value, Type, DType


class BlockBuilder:
    def __init__(self, ir_builder: 'IRBuilder', args: List[Value]):
        self.builder = ir_builder
        self.args = args

    def __enter__(self):
        self.builder.cur_block.append([])
        block = ir.Block(self.args, self.builder.cur_block[-1])
        return block

    def __exit__(self, exc_type, exc_value, tb):
        self.builder.cur_block.pop()


class IRBuilder:
    def __init__(self) -> None:
        self.cur_block: List[List[ir.Operation]] = [[]]
        # ensures every operation added is unique, id(op)
        self.op_set: Set[int] = set()
        # ensures functions do not recurse
        self.fn_scope: List[Tuple[int, str]] = []  # (id, name)

    def add_op(self, op: ir.Operation):
        if id(op) in self.op_set:
            raise RuntimeError("op is already inserted")
        self.cur_block[-1].append(op)
        self.op_set.add(id(op))

    def add_global_op(self, op: ir.Operation):
        if id(op) in self.op_set:
            raise RuntimeError("op is already inserted")
        self.cur_block[0].append(op)
        self.op_set.add(id(op))

    def new_scope(self, args: List[Value]) -> BlockBuilder:
        return BlockBuilder(self, args)

    def build_fn(self, name: str, args: List['Var'], f: Callable):
        if (id(f), name) in self.fn_scope:
            raise RuntimeError("recursion is not supported yet")
        self.fn_scope.append((id(f), name))
        with self.new_scope([a.value for a in args]) as block:
            ret = f(*args)
        self.fn_scope.pop()
        if isinstance(ret, Var):
            ret = [ret]
        block.ops.append(ir.YieldOp([v.value for v in ret]))
        fn = ir.IRFunctionOp(block, name)
        self.add_global_op(fn)

    def ir_module(self) -> ir.IRModuleOp:
        assert len(self.cur_block) == 1
        block = self.cur_block.pop()
        return ir.IRModuleOp(ir.Block([], block))


class Trace:
    builders: ClassVar[List[IRBuilder]] = []

    def __init__(self) -> None:
        self.builder = IRBuilder()

    def __enter__(self) -> IRBuilder:
        Trace.builders.append(self.builder)
        return self.builder

    def __exit__(self, exc_type, exc_value, tb):
        b = Trace.builders.pop()
        assert b is self.builder

    @staticmethod
    def current_builder() -> IRBuilder:
        assert len(Trace.builders) > 0, "builder not initialized"
        return Trace.builders[-1]


def constant(c: Union[int, float, bool]) -> 'Var':
    op = ir.ConstantOp(c)
    Trace.current_builder().add_op(op)
    return Var(op.ret[0])


NumExpr = Union[int, float, bool, 'Var']


def wrap_constexpr(v: NumExpr) -> 'Var':
    if isinstance(v, (int, float, bool)):
        return constant(v)
    return v


def var(type: Type, name: Optional[str] = None) -> 'Var':
    return Var(Value(type, name))


def ivar(name: Optional[str] = None) -> 'Var':
    return var(ir.i32, name)


def tensor(shape: Tuple[Optional[int], ...], dtype: ir.DType, name: Optional[str] = None):
    return var(ir.TensorType(dtype, shape), name)


class Var:
    def __init__(self, value: Value):
        self.value = value

    @staticmethod
    def from_type(type: ir.Type, name: Optional[str] = None):
        return Var(Value(type, name))

    @staticmethod
    def handle_binary(op_code: ir.BinaryOpCodes, lhs_: 'Var', rhs_: NumExpr) -> 'Var':
        rhs = wrap_constexpr(rhs_).value
        lhs = lhs_.value
        assert isinstance(lhs.type, ir.DType) and isinstance(rhs.type, ir.DType)
        assert op_code.type_check(lhs.type, rhs.type)
        op = ir.BinaryOp(op_code, lhs, rhs, Value(op_code.ret_ty(lhs.type, rhs.type)))
        Trace.current_builder().add_op(op)
        return Var(op.ret[0])

    def __add__(self, other: NumExpr):
        return Var.handle_binary(ir.BinaryOpCodes.Add, self, other)

    def __sub__(self, other: NumExpr):
        return Var.handle_binary(ir.BinaryOpCodes.Add, self, wrap_constexpr(other).__neg__())

    def __mul__(self, other: NumExpr):
        return Var.handle_binary(ir.BinaryOpCodes.Mul, self, other)

    def __truediv__(self, other: NumExpr):
        return Var.handle_binary(ir.BinaryOpCodes.Div, self, other)

    def __floordiv__(self, other: NumExpr):
        return Var.handle_binary(ir.BinaryOpCodes.Div, self, other)

    def __mod__(self, other: NumExpr):
        return Var.handle_binary(ir.BinaryOpCodes.Mod, self, other)

    def __eq__(self, other):
        assert isinstance(other, (int, float, bool, Var))
        return Var.handle_binary(ir.BinaryOpCodes.Eq, self, other)

    # def __bool__(self):
    #     assert False, 'Cannot convert to bool, use boolean operators instead'
    #     return False

    def __neg__(self):
        val = self.value
        ty = val.type
        assert isinstance(ty, DType) and (ty.is_floating() or ty.is_integral())
        op = ir.NegOp(val, Value(ty))
        Trace.current_builder().add_op(op)
        return Var(op.ret[0])

    def __getitem__(self, indices: Union[NumExpr, Tuple[NumExpr, ...]]):
        assert isinstance(self.value.type, ir.TensorType)
        idx = (
            [wrap_constexpr(v).value for v in indices]
            if isinstance(indices, tuple)
            else [wrap_constexpr(indices).value]
        )
        op = ir.TensorIndexOp(self.value, tuple(idx), Value(self.value.type.dtype))
        Trace.current_builder().add_op(op)
        return Var(op.ret[0])


def grid_compute(shape: Tuple[NumExpr, ...], f: Callable) -> Var:
    assert all(isinstance(s, int) for s in shape if not isinstance(s, Var))
    builder = Trace.current_builder()
    args = [var(ir.i32) for _ in range(len(shape))]
    new_shape = tuple(wrap_constexpr(s).value for s in shape)
    with builder.new_scope([a.value for a in args]) as block:
        ret = f(*args)
    assert isinstance(ret, Var)
    assert isinstance(ret.value.type, ir.DType)
    block.ops.append(ir.YieldOp([ret.value]))
    ret_tensor_shape = tuple(s if isinstance(s, int) else None for s in shape)
    ret_tensor_type = ir.TensorType(ret.value.type, ret_tensor_shape)
    op = ir.GridComputeOp(new_shape, block, Value(ret_tensor_type))
    builder.add_op(op)
    return Var(op.ret[0])


def grid_reduce(reduction: ir.ReduceOperations, shape: Tuple[NumExpr, ...], f: Callable) -> Var:
    assert all(isinstance(s, int) for s in shape if not isinstance(s, Var))
    builder = Trace.current_builder()
    args = [var(ir.i32) for _ in range(len(shape))]
    new_shape = tuple(wrap_constexpr(s).value for s in shape)
    with builder.new_scope([a.value for a in args]) as block:
        ret = f(*args)
    assert isinstance(ret, Var)
    assert isinstance(ret.value.type, ir.DType)
    block.ops.append(ir.YieldOp([ret.value]))
    op = ir.GridReduceOp(reduction, new_shape, block, Value(ret.value.type))
    builder.add_op(op)
    return Var(op.ret[0])


def math_fintrinsic(op_code: str, a: Var) -> Var:
    assert isinstance(a.value.type, DType) and a.value.type.is_floating()
    op = ir.ElementwiseMathOp(op_code, a.value)
    Trace.current_builder().add_op(op)
    return Var(op.ret[0])


def exp(a: Var) -> Var:
    return math_fintrinsic('exp', a)


def tensor_shape(tensor: Var) -> Tuple[Optional[int], ...]:
    assert isinstance(tensor.value.type, ir.TensorType)
    return tensor.value.type.shape


def shape_of(tensor: Var, idx: int) -> Union[int, Var]:
    assert isinstance(tensor.value.type, ir.TensorType)
    assert 0 <= idx < len(tensor.value.type.shape)
    s = tensor.value.type.shape[idx]
    if isinstance(s, int):
        return s
    op = ir.TensorShapeOfOp(tensor.value, idx)
    Trace.current_builder().add_op(op)
    return Var(op.ret[0])


def sassert(cond: NumExpr):
    if not isinstance(cond, Var):
        assert isinstance(cond, bool) and cond
        return
    res = wrap_constexpr(cond)
    Trace.current_builder().add_op(ir.AssertOp(res.value))
