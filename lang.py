# %%
from typing import Callable, Union
import ir
from ir import Type


class Expr:
    """
    An Expr could also be any python value that ir.is_base_value returns True.
    """
    ValidExpr = Union['Expr', int, float, bool, str, tuple['ValidExpr']]

    def __init__(self, op_code: str, args: tuple[ValidExpr]):
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
    
    def __str__(self):
        if ir.is_base_value(self):
            return str(self)
        if hasattr(self, '__str__') and type(self) != Expr:
            return str(self)
        if len(self.args) == 0:
            return f'{self.op_code}'
        elif len(self.args) == 1:
            return f'{self.op_code}({self.args[0]})'
        else:
            return f'{self.op_code}({", ".join(map(str, self.args))})'
        

class Variable(Expr):
    def __init__(self, name: str, type: Type):
        super().__init__('var', ())
        self.name = name
        self.type = type
    
    def __str__(self):
        return f'{self.name}: {self.type}'


class GridCompute(Expr):
    def __init__(self, shape: tuple['Expr'], f: Callable):
        super().__init__('grid', shape)
        self.shape = shape
        self.args = [Variable(f'i{i}', ir.i32) for i in range(len(shape))]
        self.expr = f(*self.args)
    
    def __str__(self):
        return 'grid(shape={}) |{}|'.format(
            ', '.join(map(str, self.shape)),
            ', '.join(map(str, self.args))
        ) + '{ ' + str(self.expr) + ' }'

class GridReduce(Expr):
    def __init__(self, shape: tuple['Expr'], op: ir.ReduceOperations, f: Callable):
        super().__init__('grid', shape)
        self.shape = shape
        self.args = [Variable(f'i{i}', ir.i32) for i in range(len(shape))]
        self.expr = f(*self.args)
        self.op = op
    
    def __str__(self):
        return 'reduce(shape={}) |{}|'.format(
            ', '.join(map(str, self.shape)),
            ', '.join(map(str, self.args))
        ) + ' { ' + str(self.expr) + ' } '

m = Variable('m', ir.i32)
n = Variable('n', ir.i32)
k = Variable('k', ir.i32)

a = Variable('A', ir.TensorType(ir.f32, (m, k)))
b = Variable('B', ir.TensorType(ir.f32, (k, n)))
c = GridReduce((m, n), ir.ReduceOperations.Sum, lambda i, j: a[i, k] * b[k, j])

print(c)
