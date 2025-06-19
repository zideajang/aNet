from functools import partialmethod
import numpy as np
from rich.console import Console
import torch
console = Console()

class Tensor:
    def __init__(self,data:np.ndarray):

        if type(data) != np.ndarray:
            console.print(f"需要 np 类型数据:{data}")

        self.data:np.ndarray = data
        self.grad:np.ndarray | None = None

        self._ctx:Context | None = None
    def __str__(self):
        return f"shape of Tensor:{self.data.shape} with grad {self.grad}"

    def backward(self,allow_fill:bool=True):
        
        if self._ctx is None:
            return 
        
        if self.grad is None and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert (self.grad is not None)
        grads = self._ctx.arg.backward(self._ctx,self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        
        # 也就是根据当前
        for t, g in zip(self._ctx.parents,grads):
            if g.shape != t.data.shape:
                console.print(f"grad shape must match tensor shape in {g.shape}, {t.data.shape}")
                assert(False)
            t.grad = g
            t.backward(False)
    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)



class Context:
    def __init__(self,arg:'Function',*tensors):
        self.arg = arg
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self,*x):
        self.saved_tensors.extend(x)


class Function:
    def apply(self,arg,*x):
        ctx = Context(arg,self,*x)
        ret = Tensor(arg.forward(ctx,self.data,*[t.data for t in x]))
        ret._ctx = ctx
        return ret
    


def register(name,fxn):
    setattr(Tensor,name,partialmethod(fxn.apply,fxn))


class Mul(Function):
    @staticmethod
    def forward(ctx:Context,x,y):
        ctx.save_for_backward(x,y)
        print(type(x))
        return x*y
    
    @staticmethod
    def backward(ctx:Context,grad_output):
        x,y = ctx.saved_tensors
        return y*grad_output, x*grad_output

register('mul',Mul)

# 激活函数 ReLU
class ReLU(Function):
    @staticmethod
    def forward(ctx:Context,input:np.ndarray):
        ctx.save_for_backward(input)
        return np.maximum(input,0)
    
    @staticmethod
    def backward(ctx:Context,grad_output:np.ndarray):
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
    
register('relu',ReLU)

class Dot(Function):
    @staticmethod
    def forward(ctx:Context,input,weight):
        ctx.save_for_backward(input,weight)
        return input.dot(weight)
    
    @staticmethod
    def backward(ctx:Context,grad_output):
        input,weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input,grad_weight

register('dot',Dot)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return x+y

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add)

class Sum(Function):
    @staticmethod
    def forward(ctx:Context,input:np.ndarray):
        ctx.save_for_backward(input)
        return np.array([input.sum()])
    
    @staticmethod
    def backward(ctx:Context,grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)

register('sum',Sum)

# 逻辑层函数
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx:Context,input:np.ndarray):

        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x - c.reshape((-1,1))).sum(axis=1))
        output = input - logsumexp(input).reshape((-1,1))
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx:Context,grad_output):
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1,1))
register('logsoftmax',LogSoftmax)
    
    

if __name__ == "__main__":
    a = Tensor(data=np.array([1.0,2.0,3.0]))
    a_sum = a.sum()
    console.print(a_sum.data)
    a_sum.backward()


    x_tensor = Tensor(data=np.array([2.0,3.0]))
    y_tensor = Tensor(data=np.array([1.0,1.0]))

    z_tensor = x_tensor.dot(y_tensor)
    print(z_tensor.data)
    z_tensor.backward()
    print(x_tensor.grad)
    print(y_tensor.grad)

    print('---'*20)
    print("\n")
    x_tensor_torch = torch.tensor(np.array([2.0, 3.0]), dtype=torch.float32, requires_grad=True)
    y_tensor_torch = torch.tensor(np.array([1.0, 1.0]), dtype=torch.float32, requires_grad=True)

    # Perform the dot product
    z_tensor_torch = torch.dot(x_tensor_torch, y_tensor_torch)

    # Print the data (equivalent to z_tensor.data)
    print(z_tensor_torch.data)

    # Perform the backward pass
    z_tensor_torch.backward()

    # Print the gradients (equivalent to x_tensor.grad and y_tensor.grad)
    print(x_tensor_torch.grad)
    print(y_tensor_torch.grad)