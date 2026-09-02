# building micrograd
# first find a way to represent scalar values

class Value:
    def __init__(self, data,children=(),op=''):
        self.data = data
        self.op = op
        self._prev = set(children)
        self.grad = 0.0
    
    def __repr__(self):
        return f"Value={self.data}"

    def __add__(self, other):
        out = Value(self.data + other.data, (self,other),'+')

        def _backward():
            self.grad += out.grad
            other .grad+= out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self,other),'*')

        def _backward():
            self.grad+= out.grad*other.data
            other.grad+=out.grad*self.data
        out._backward = _backward
        return out
        

    def __pow__(self,other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad+=out.grad*other*self.data**(other-1)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad+=out.grad*(1 if self.data>0 else 0)
        out._backward=_backward
        return out

a = Value(-0.06)
b = Value(3.0)
# print(a.data)

c = a+b
print(c)
# print(a.relu())