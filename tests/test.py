
import numpy as np

import torch
import torch.nn.functional as F

from rich.console import Console
from rich.panel import Panel

from anet import Tensor

x_init = np.random.randn(5,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)

b_init = np.random.randn(1,3).astype(np.float32)

# softmax(relu(x@w)).sum()


console = Console()
console.print(Panel("测试算子",title="测试 aNet 算子"))
x_torch_tensor = torch.tensor(x_init,requires_grad=True)
w_torch_tensor = torch.tensor(W_init,requires_grad=True)
b_torch_tensor = torch.tensor(b_init)

z_torch_tensor = x_torch_tensor @ w_torch_tensor
z_torch_tensor.relu()
out_torch_logsoftmax_tensor = torch.nn.functional.log_softmax(z_torch_tensor, dim=1)

out_torch_sum_tensor = out_torch_logsoftmax_tensor.sum()

out_torch_sum_tensor.backward()
console.print("----------- Torch -------------")
console.print(out_torch_sum_tensor.detach().numpy())
console.print(x_torch_tensor.grad) 
console.print(w_torch_tensor.grad)


x_tensor = Tensor(x_init)
w_tensor = Tensor(W_init)
b_tensor = Tensor(b_init)

z_tensor = x_tensor.dot(w_tensor) 
z_tensor.relu()
out_logsoftmax_tensor = z_tensor.logsoftmax()
out_sum_tensor = out_logsoftmax_tensor.sum()
out_sum_tensor.backward()

console.print("----------- aNet -------------")
console.print(out_sum_tensor.data)
console.print(x_tensor.grad) 
console.print(w_tensor.grad)

