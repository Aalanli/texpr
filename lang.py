# %%
from typing import List, Set, Tuple, Callable, Dict, Optional, Union
import ir
from ir import Type
import utils

ValidExpr = Union['Expr', int, str, float, bool, tuple['ValidExpr']]

class Expr:
    """
    An Expr could also be any python value that ir.is_base_value returns True.
    """

    def __init__(self, op_code: str, args: tuple[ValidExpr,...]):
        self.op_code = op_code
        self.args = args

    def __add__(self, other: ValidExpr):
        return Expr('add', (self, other))

    def __sub__(self, other: ValidExpr):
        return Expr('sub', (self, other))

    def __mul__(self, other: ValidExpr):
        return Expr('mul', (self, other))

    def __truediv__(self, other: ValidExpr):
        return Expr('true_div', (self, other))

    def __floordiv__(self, other: ValidExpr):
        return Expr('floor_div', (self, other))

    def __mod__(self, other: ValidExpr):
        return Expr('mod', (self, other))

    def __bool__(self):
        raise RuntimeError('Cannot convert to bool, use boolean operators instead')

    def __not__(self):
        return Expr('not', (self,))

    def __neg__(self):
        return Expr('neg', (self,))

    def logical_and(self, other: ValidExpr):
        return Expr('logical_and', (self, other))

    def logical_or(self, other: ValidExpr):
        return Expr('logical_or', (self, other))

    def if_then_else(self, true_value: ValidExpr, false_value: ValidExpr):
        return Expr('if_then_else', (self, true_value, false_value))
    
    def __getitem__(self, indices):
        return Expr('get_item', (self, indices))
    
    @staticmethod
    def to_str(self):
        if isinstance(self, (str, int, float, bool)):
            return str(self)
        elif isinstance(self, Variable):
            return f'{self.name}: {self.type}'
        elif isinstance(self, Expr):
            if len(self.args) == 0:
                return f'{self.op_code}'
            elif len(self.args) == 1:
                return f'{self.op_code}({self.args[0]})'
            else:
                return f'{self.op_code}({", ".join(map(Expr.to_str, self.args))})'
        elif isinstance(self, tuple):
            return f'(' + ', '.join(map(Expr.to_str, self)) + ')'
            
        raise RuntimeError(f"unexpected value {self}")
        
    def __str__(self):
        return self.to_str(self)

class Variable(Expr):
    uid = 0
    def __init__(self, type: Type, name: Optional[str] = None):
        super().__init__('var', ())
        self.name = name if name is not None else f'%{Variable.uid}'
        if name is None:
            Variable.uid += 1
        self.type = type
        self.associated_value = ir.Value(type)
    
    def __str__(self):
        return f'{self.name}: {self.type}'
    
    def __hash__(self):
        return id(self)

class GridCompute(Expr):
    def __init__(self, shape: tuple[ValidExpr, ...], f: Callable):
        super().__init__('grid_compute', shape)
        self.shape = shape
        self.bargs = [Variable(ir.i32, f'i{i}') for i in range(len(shape))]
        self.expr = f(*self.bargs)
    
    def __str__(self):
        return 'grid(shape={}) |{}|'.format(
            ', '.join(map(str, self.shape)),
            ', '.join(map(str, self.bargs))
        ) + '{\n ' + str(self.expr) + '\n}'

class GridReduce(Expr):
    def __init__(self, shape: Tuple[ValidExpr,...], op: ir.ReduceOperations, f: Callable):
        super().__init__('grid_reduce', shape)
        self.shape = shape
        self.bargs = [Variable(ir.i32, f'i{i}') for i in range(len(shape))]
        self.expr = f(*self.bargs)
        self.op = op
    
    def __str__(self):
        return 'reduce(shape={}) |{}|'.format(
            ', '.join(map(str, self.shape)),
            ', '.join(map(str, self.bargs))
        ) + ' { \n' + str(self.expr) + '\n} '


def var(type: ir.Type, name: Optional[str] = None) -> Variable:
    return Variable(type, name)

def ivar(name: Optional[str] = None) -> Variable:
    return Variable(ir.i32, name)

def tensor(shape: Tuple[Union[int, Variable],...], dtype: ir.DType, name: Optional[str] = None) -> Variable:
    new_shape: List[Union[int, ir.Value]] = []
    for s in shape:
        if isinstance(s, int):
            new_shape.append(s)
        elif isinstance(s, Variable):
            new_shape.append(s.associated_value)
    tensor_ty = ir.TensorType(dtype, tuple(new_shape))
    return var(tensor_ty, name)

def grid_compute(shape: Tuple[Union[int, Variable],...], f: Callable) -> GridCompute:
    return GridCompute(shape, f)

def reduce_compute(shape: Tuple[Union[int, Variable],...], f: Callable, op: ir.ReduceOperations = ir.ReduceOperations.Sum) -> GridReduce:
    return GridReduce(shape, op, f)

def shape_of(tensor: Expr, index: int):
    return Expr('shape_of', (tensor, index))

class ExprVisitor:
    def __init__(self) -> None:
        self.cur_block: List[List[ir.Operation]] = [[]]
        self.defined_vals: List[Set[ir.Value]] = [set()]
        self.var_table: Set[Variable] = set()
        self.memo: Dict[int, ir.Value] = {}

    def append_op(self, op):
        self.cur_block[-1].append(op)
    
    def new_block(self):
        self.cur_block.append([])
        self.defined_vals.append(set())
    
    def pop_block(self):
        self.defined_vals.pop()
        return self.cur_block.pop()
    
    def is_val_defined(self, val: ir.Value) -> bool:
        for i in range(len(self.defined_vals) - 1, -1, -1):
            if val in self.defined_vals[i]:
                return True
        return False
    
    def define_val(self, val: ir.Value):
        self.defined_vals[-1].add(val)
    
    def register_var(self, var: Variable) -> ir.Value:
        if var not in self.var_table:
            self.var_table.add(var)
        return var.associated_value

    def get_value(self, var: Variable) -> ir.Value:
        if var not in self.var_table:
            raise RuntimeError("cannot fetch var {}".format(var))
        return var.associated_value
    
    def visit(self, x: ValidExpr) -> ir.Value:
        uid = id(x)
        # should be safe to memorize,
        # if it satisfies python scoping rules, it satisfies IR dominance rules
        if uid in self.memo and self.is_val_defined(self.memo[uid]):
            return self.memo[uid]
        if isinstance(x, (int, bool, float)):
            op = ir.ConstantOp(x)
            self.append_op(op)
            res = op.ret[0]
        assert isinstance(x, Expr)
        vname = 'visit_' + x.op_code
        if hasattr(self, vname):
            f = getattr(self, vname)
            assert callable(f)
            res = f(x)
            assert isinstance(res, ir.Value)
        else:        
            raise NotImplementedError(f'visitor for {x.op_code} is not implemented')
        
        self.memo[uid] = res
        self.define_val(res)
        return res

    def handle_binary(self, opcode: ir.BinaryOpCodes, lhs: ir.Value, rhs: ir.Value, res_type = None):
        assert lhs.type == rhs.type
        assert isinstance(lhs.type, ir.DType)
        if res_type is None:
            if opcode.value >= ir.BinaryOpCodes.Eq.value:
                res_type = ir.i1
            else:
                res_type = lhs.type
        ret = ir.Value(res_type)
        op = ir.BinaryOp(opcode, lhs, rhs, ret)
        self.append_op(op)
        return op.ret[0]

    def visit_add(self, x: Expr):
        return self.handle_binary(ir.BinaryOpCodes.Add, self.visit(x.args[0]), self.visit(x.args[1]))
    
    def visit_sub(self, x: Expr):
        lhs = self.visit(x.args[1])
        op = ir.NegOp(lhs, ir.Value(lhs.type))
        self.append_op(op)
        return self.handle_binary(ir.BinaryOpCodes.Add, self.visit(x.args[0]), op.ret[0])
    
    def visit_mul(self, x: Expr):
        return self.handle_binary(ir.BinaryOpCodes.Mul, self.visit(x.args[0]), self.visit(x.args[1]))
    
    def visit_true_div(self, x: Expr):
        return self.handle_binary(ir.BinaryOpCodes.Div, self.visit(x.args[0]), self.visit(x.args[1]), res_type=ir.f32)
    
    def visit_floor_div(self, x: Expr):
        lhs: ir.Value = self.visit(x.args[0])
        assert lhs.type == ir.i32
        return self.handle_binary(ir.BinaryOpCodes.Div, lhs, self.visit(x.args[1]))
    
    def visit_mod(self, x: Expr):
        lhs = self.visit(x.args[0])
        assert lhs.type == ir.i32
        return self.handle_binary(ir.BinaryOpCodes.Mod, lhs, self.visit(x.args[1]))
    
    def visit_neg(self, x: Expr):
        v = self.visit(x.args[0])
        op = ir.NegOp(v, ir.Value(v.type))
        self.append_op(op)
        return op.ret[0]
    
    def visit_not(self, x: Expr):
        v = self.visit(x.args[0])
        op = ir.NotOp(v, ir.Value(v.type))
        self.append_op(op)
        return op.ret[0]

    def visit_logical_and(self, x: Expr):
        lhs = self.visit(x.args[0])
        rhs = self.visit(x.args[1])
        assert lhs.type == ir.i1
        return self.handle_binary(ir.BinaryOpCodes.And, lhs, rhs)
    
    def visit_logical_or(self, x: Expr):
        lhs = self.visit(x.args[0])
        rhs = self.visit(x.args[1])
        assert lhs.type == ir.i1
        return self.handle_binary(ir.BinaryOpCodes.Or, lhs, rhs)
    
    def visit_if_then_else(self, x: Expr):
        cond = self.visit(x.args[0])
        assert cond.type == ir.i1
        then_val = self.visit(x.args[1])
        else_val = self.visit(x.args[2])
        assert then_val.type == else_val.type
        op = ir.IfThenElseOp(cond, then_val, else_val, ir.Value(then_val.type))
        self.append_op(op)
        return op.ret[0]

    def visit_get_item(self, x: Expr):
        a = self.visit(x.args[0])
        assert isinstance(a, ir.Value) and isinstance(a.type, ir.TensorType)
        if isinstance(x.args[1], tuple):
            indices = tuple(self.visit(i) for i in x.args[1])
        else:
            indices = (self.visit(x.args[1]),)
        op = ir.TensorIndexOp(a, indices, ir.Value(a.type.dtype))
        self.append_op(op)
        return op.ret[0]
    
    def visit_var(self, x: Variable):
        return self.get_value(x)
    
    def visit_grid_compute(self, x: GridCompute):
        shape = tuple(self.visit(s) for s in x.shape)
        self.new_block()
        block_args = [self.register_var(v) for v in x.bargs]
        res = self.visit(x.expr)
        # for now, only support scalar returns
        assert isinstance(res, ir.Value) and isinstance(res.type, ir.DType)

        block_ops = self.pop_block()
        block_ops.append(ir.YieldOp(res))
        block = ir.Block(block_args, block_ops)

        ret_ty = ir.TensorType(res.type, shape)

        op = ir.GridComputeOp(shape, block, ir.Value(ret_ty))
        self.append_op(op)
        return op.ret[0]

    def visit_grid_reduce(self, x: GridReduce):
        shape = tuple(self.visit(s) for s in x.shape)
        self.new_block()
        block_args = [self.register_var(v) for v in x.bargs]
        res = self.visit(x.expr)
        block_ops = self.pop_block()
        block_ops.append(ir.YieldOp(res))
        block = ir.Block(block_args, block_ops)

        # for now, only support scalar returns
        assert isinstance(res, ir.Value) and isinstance(res.type, ir.DType)

        op = ir.GridReduceOp(x.op, shape, block, ir.Value(res.type))
        self.append_op(op)
        return op.ret[0]
    
    def visit_shape_of(self, x: Expr):
        res = self.visit(x.args[0])
        assert isinstance(res.type, ir.TensorType)
        assert isinstance(x.args[1], int)
        sh = res.type.shape[x.args[1]]
        if isinstance(sh, int):
            op1 = ir.ConstantOp(sh)
            self.append_op(op1)
            sh1 = op1.ret[0]
        else:
            # prevent variable escape
            op2 = ir.TensorShapeOfOp(res, x.args[1])
            self.append_op(op2)
            sh1 = op2.ret[0]
        return sh1
    
    def visit_math_elementwise(self, x: Expr):
        res = self.visit(x.args[1])
        assert isinstance(x.args[0], str)
        op = ir.ElementwiseMathOp(x.args[0], res)
        self.append_op(op)
        return op.ret[0]

def math_elementwise(x: Expr, opcode: str) -> Expr:
    return Expr("math_elementwise", (opcode, x))

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
