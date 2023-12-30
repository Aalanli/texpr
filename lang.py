# %%
from typing import List, Set, Tuple, Callable, Dict, Optional, Union
import ir
from ir import Type
import utils

NumExpr = Union['Expr', int, float, bool]
TraceableExpr = Union[NumExpr, Tuple['TraceableExpr',...]]
TraceResult = Union[ir.Value, Tuple['TraceResult',...]]

class BlockBuilder:
    def __init__(self, ir_builder: 'IRBuilder'):
        self.builder = ir_builder
    
    def __enter__(self):
        self.builder.cur_block.append([])
        self.builder.defined_vals.append(set())
        self.builder.scope_map.append(dict())
        return self.builder.cur_block[-1]
    
    def __exit__(self):
        self.builder.cur_block.pop()
        self.builder.defined_vals.pop()
        self.builder.scope_map.pop()
    
class IRBuilder:
    def __init__(self):
        self.cur_block: List[List[ir.Operation]] = [[]]
        self.defined_vals: List[Set[ir.Value]] = [set()]
        self.scope_map: List[Dict[int, Union[ir.Operation, ir.Value]]] = []

    def add_op(self, op):
        self.cur_block[-1].append(op)

    def new_scope(self) -> BlockBuilder:
        return BlockBuilder(self)
    
    def is_val_defined(self, val: ir.Value) -> bool:
        for i in range(len(self.defined_vals) - 1, -1, -1):
            if val in self.defined_vals[i]:
                return True
        return False
    
    def define_val(self, val: ir.Value):
        self.defined_vals[-1].add(val)
    
    def trace_on_current_scope(self, expr: NumExpr):
        if self.get_on_current_scope(expr) is not None:
            return
        if isinstance(expr, (int, float, bool)):
            op = ir.ConstantOp(expr)
            self.add_op(op)
            self.scope_map[-1][id(expr)] = op
        else:
            if expr.op_code in ('var', 'tensor'):
                # terminals, TODO: make this better
                res = expr.handler(self, [])
                assert len(expr.intrace) == 0
                assert len(res) == 1 and isinstance(res[0], ir.Value)
                self.scope_map[-1][id(expr)] = res[0]
                return
            for sub_expr in expr.intrace:
                self.trace_on_current_scope(sub_expr)
            
    def get_on_current_scope(self, expr: NumExpr) -> Optional[Union[ir.Value, ir.Operation]]:
        for i in range(len(self.scope_map) -1, -1, -1):
            if id(expr) in self.scope_map[i]:
                return self.scope_map[id(expr)]
        return None

class Expr:
    def __init__(self,
                 op_code: str,
                 intrace: List[NumExpr], outputs: List[ir.Type], 
                 handler: Callable[[IRBuilder, List[ir.Value]], List[ir.Value]]):
        self.op_code = op_code
        self.intrace = intrace
        self.outputs = outputs
        self.handler = handler
    
    @staticmethod
    def handle_binary(code: ir.BinaryOpCodes, lhs: 'Expr', rhs: NumExpr):
        assert len(lhs.outputs) == 1
        if isinstance(rhs, Expr):
            assert len(rhs.outputs) == 1
            rty = rhs.outputs[0]
        elif isinstance(rhs, int):
            rty = ir.i32
        elif isinstance(rhs, float):
            rty = ir.f32
        else:
            rty = ir.i1
        assert code.type_check(lhs.outputs[0], rty)
        def handler(builder: IRBuilder, arg: Tuple[ir.Value,ir.Value]) -> Tuple[ir.Value]:
            op = ir.BinaryOp(code, arg[0], arg[1], ir.Value(rty))
            builder.add_op(op)
            return (op.ret[0])
        return Expr(intrace=[lhs, rhs], outputs=[rty], handler=handler)

    def __add__(self, other: Union[int, float, 'Expr']):
        return Expr.handle_binary(ir.BinaryOpCodes.Add, self, other)

    def __sub__(self, other: Union[int, float, 'Expr']):
        return Expr.handle_binary(ir.BinaryOpCodes.Add, self, other)

    def __mul__(self, other: Union[int, float, 'Expr']):
        return Expr.handle_binary(ir.BinaryOpCodes.Add, self, other)

    def __truediv__(self, other: Union[int, float, 'Expr']):
        return Expr.handle_binary(ir.BinaryOpCodes.Add, self, other)

    def __floordiv__(self, other: Union[int, float, 'Expr']):
        return Expr.handle_binary(ir.BinaryOpCodes.Add, self, other)

    def __mod__(self, other: Union[int, float, 'Expr']):
        return Expr.handle_binary(ir.BinaryOpCodes.Add, self, other)

    def __bool__(self):
        raise RuntimeError('Cannot convert to bool, use boolean operators instead')

    def __neg__(self):
        assert len(self.outputs) == 1
        ty = self.outputs[0]
        assert isinstance(ty, ir.DType) and (ty.is_floating() or ty.is_integral())
        def handler(builder: IRBuilder, a: Tuple[ir.Value]) -> Tuple[ir.Value]:
            op = ir.NegOp(a[0], ir.Value(a[0].type))
            builder.add_op(op)
            return op.ret[-1]
        return Expr([self], [ty], handler)

    def __not__(self):
        assert len(self.outputs) == 1
        ty = self.outputs[0]
        assert isinstance(ty, ir.DType) and (ty.is_integral() or ty == ir.i1)
        def handler(builder: IRBuilder, a: Tuple[ir.Value]) -> Tuple[ir.Value]:
            op = ir.NotOp(a[0], ir.Value(a[0].type))
            builder.add_op(op)
            return op.ret[-1]
        return Expr([self], [ty], handler)

    # def logical_and(self, other: ValidExpr):
    #     raise NotImplementedError()
    # def logical_or(self, other: ValidExpr):
    #     raise NotImplementedError()

    # def if_then_else(self, true_value: ValidExpr, false_value: ValidExpr):
    #     raise NotImplementedError()
        
    def __getitem__(self, indices: Union[NumExpr, Tuple[NumExpr,...]]):
        return Expr('get_item', (self, indices))

def numexpr_dtype(expr: NumExpr) -> ir.DType:
    if isinstance(expr, int):
        return ir.i32
    elif isinstance(expr, float):
        return ir.f32
    elif isinstance(expr, bool):
        return ir.i1
    else:
        assert len(expr.outputs) == 1 and isinstance(expr.outputs[0], ir.DType)
        return expr.outputs[0]

def var(type: ir.Type, name: Optional[str] = None) -> Expr:
    return Expr('var', [], [type], lambda b, a: ir.Value(type, name_hint=name))

def ivar(name: Optional[str] = None) -> Expr:
    return var(ir.i32, name)

def tensor(shape: Tuple[Optional[int],...], dtype: ir.DType, name: Optional[str] = None) -> Expr:
    ty = ir.TensorType(dtype, shape)
    return Expr('tensor', [], [ty], lambda b, a: ir.Value(ty, name))

def grid_compute(shape: Tuple[NumExpr,...], f: Callable) -> Expr:
    assert all(len(i.outputs) == 1 and i.outputs[0] == ir.i32 for i in filter(lambda x: isinstance(x, Expr), shape))
    block_args = [ivar() for _ in range(len(shape))]
    res = f(*block_args)
    assert isinstance(res, NumExpr)
    ty = ir.TensorType(numexpr_dtype(res), shape=tuple(i if isinstance(i, int) else None for i in shape))
    def handler(builder: IRBuilder, inputs: List[ir.Value]) -> List[ir.Value]:
        with builder.new_scope() as block_ops:
            builder.trace_on_current_scope(res)
            args: List[ir.Value] = [builder.get_on_current_scope(a) for a in block_args]
        block = ir.Block(args, block_ops)
        op = ir.GridComputeOp(inputs, block, ir.Value(ty))
        builder.add_op(op)
        return [op.ret[0]]

    Expr('grid_compute', intrace=shape, outputs=[ty], handler=handler)

def reduce_compute(shape: Tuple[NumExpr,...], f: Callable, reduction: ir.ReduceOperations = ir.ReduceOperations.Sum) -> Expr:
    assert all(len(i.outputs) == 1 and i.outputs[0] == ir.i32 for i in filter(lambda x: isinstance(x, Expr), shape))
    block_args = [ivar() for _ in range(len(shape))]
    res = f(*block_args)
    assert isinstance(res, NumExpr)
    ty = numexpr_dtype(res)
    def handler(builder: IRBuilder, inputs: List[ir.Value]) -> List[ir.Value]:
        with builder.new_scope() as block_ops:
            builder.trace_on_current_scope(res)
            args: List[ir.Value] = [builder.get_on_current_scope(a) for a in block_args]
        block = ir.Block(args, block_ops)
        op = ir.GridReduceOp(reduction, inputs, block, ir.Value(ty))
        builder.add_op(op)
        return [op.ret[0]]

    Expr('grid_reduce', intrace=shape, outputs=[ty], handler=handler)

def shape_of(tensor: Expr, index: int):
    assert len(tensor.outputs) == 1 and isinstance(tensor.outputs[0], ir.TensorType), "type error"
    ty = tensor.outputs[0]
    def handler(builder: IRBuilder, inputs: List[ir.Value]) -> List[ir.Value]:
        op = ir.TensorShapeOfOp(inputs[0], index)
        builder.add_op(op)
        return op.ret[0]
    return Expr('shape_of', [tensor], [ty.dtype], handler)

def math_elementwise(x: Expr, opcode: str) -> Expr:    
    assert len(x.outputs) == 1 and isinstance(x.outputs[0], ir.DType), "type error"
    ty = x.outputs[0]
    def handler(builder: IRBuilder, inputs: List[ir.Value]) -> List[ir.Value]:
        op = ir.ElementwiseMathOp(opcode, inputs[0])
        builder.add_op(op)
        return op.ret[0]
    return Expr('shape_of', [x], [ty], handler)

def exp(x: Expr) -> Expr:
    return math_elementwise(x, "exp")

def trace_function(args: List[Variable], root: Expr, name: str) -> ir.IRFunctionOp:
    visitor = ExprVisitor()
    vargs = [visitor.register_var(a) for a in args]
    res = visitor.visit(root)
    assert isinstance(res, ir.Value)

    body = visitor.pop_block()
    body.append(ir.YieldOp(res))
    block = ir.Block(vargs, body)
    op = ir.IRFunctionOp(block, name)
    return op

def matmul():
    m = ivar('m')
    k = ivar('k')
    n = 3

    A = tensor((m, k), ir.f32)
    B = tensor((k, n), ir.f32)
    C = grid_compute((shape_of(A, 0), shape_of(B, 1)), 
                    lambda i, j: reduce_compute((shape_of(A, 1),),
                                                lambda k: A[i, k] * B[k, j]))
    fn = trace_function([A, B], C, 'matmul')
    return fn
import time

it = time.time()
fn = matmul()
# print(ir.dump_ir(fn))

n = ivar('n')
A = tensor((n,), ir.f32)
n1 = shape_of(A, 0)
maxes = reduce_compute((n1,),
                       lambda i: A[i], ir.ReduceOperations.Max)
a1 = grid_compute((n1,), 
                  lambda i: A[i] - maxes)
a2 = grid_compute((n1,),
                  lambda i: exp(a1[i]))
s = reduce_compute((n1,),
                   lambda i: a2[i])
y = grid_compute((n1,),
                 lambda i: a2[i] / s)

fn = trace_function([A], y, 'softmax')
print(ir.basic_verify_ir(fn))

ir.PrettyRenameValues().run_on_op(fn)

print(ir.basic_verify_ir(fn))

# print(ir.dump_ir(fn))
ir.PureLICM().run_on_op(fn)
print(ir.basic_verify_ir(fn))
# print(ir.dump_ir(fn))

capture = ir.CollectPureOpImplicitReference()
capture.run_on_op(fn)
dep = ir.OpValueDependenceAnalysis()
dep.run_on_op(fn)
ir.PureGreedyCSE(capture, dep).run_on_op(fn)
print(ir.basic_verify_ir(fn))
# print(ir.dump_ir(fn))
liveliness = ir.PureLivelinessAnalysis()
liveliness.run_on_op(fn)

# print(ir.dump_ir(fn, liveliness.alive))
ir.PureDce(liveliness).run_on_op(fn)
print(ir.basic_verify_ir(fn))
print(fn)
