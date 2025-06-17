import os
import hashlib

import gzip
import numpy as np
from anet import Tensor
from tqdm import trange

import matplotlib.pyplot as plt

from rich.console import Console
console = Console()

# 数据数据
MNIST_DATASET_PATH = "./datasets/mnist"

def fetch(url):
    fp = os.path.join(MNIST_DATASET_PATH,hashlib.md5(url.encode('utf-8')).hexdigest())
    with open(fp,"rb") as f:
        dat = f.read()

    return np.frombuffer(gzip.decompress(dat),dtype=np.uint8).copy()

X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape(-1,28,28)
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape(-1,28,28)
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


# 校验数据格式
# console.print(type(X_train))
# console.print(X_train[0x10:].reshape(-1,28,28).shape)
# console.print(Y_train[8:][0])

# 预览数据
# plt.imshow(X_train[0x10:].reshape(-1,28,28)[0])
# plt.show()

# 超参数
batch_size= 16
learning_rate = 0.01
epoch = 1000

# 定义模型
def layer_init(input_size,hidden_size):
    # 参数初始 (-1/sqrt(in*out),1/sqrt(in*out))
    ret = np.random.uniform(-1.,1.,size=(input_size,hidden_size))/np.sqrt(input_size*hidden_size)
    return ret.astype(np.float32)

# 初始化每一层的权重， y = wx
# 784 = 28x28
layer_1 = Tensor(layer_init(784,128))
layer_2 = Tensor(layer_init(128,10))

# 训练
losses,accuracies = [],[]
for i in (t:=trange(epoch)):
    # 在训练数据集中随机采样
    samp = np.random.randint(0,X_train.shape[0],size=(batch_size))

    # batch size 采样
    # 将采样数据展平为 1 (1,784)
    x = Tensor(X_train[samp].reshape((-1,28*28)))
    # ()
    Y = Y_train[samp]

    # y(128,10)
    y = np.zeros((len(samp),10),np.float32)
    # y[128,5] = -1.0 
    # 将 label 转化为one-hot编码形式 [0,0,0,-1]
    y[range(y.shape[0]),Y] = -1.0
    y = Tensor(y)

    # 全连接层
    x = x.dot(layer_1)
    # ReLU 激活层
    x = x.relu()
    # 全连接
    x = x_l2 = x.dot(layer_2)
    # logic 层
    x = x.logsoftmax()
    # xy
    # NLL Loss
    x = x.mul(y)
    # 计算loss
    x = x.mean()

    # 反向传播
    x.backward()

    loss = x.data
    cat = np.argmax(x_l2.data,axis=1)
    acc = (cat == Y).mean()

    # SGD
    layer_1.data = layer_1.data - learning_rate*layer_1.grad
    layer_2.data = layer_2.data - learning_rate*layer_2.grad

    losses.append(losses)
    accuracies.append(acc)
    t.set_description(f"loss {loss} accuracy:{acc}")


def predict(x):
    x = x.dot(layer_1.data)
    x = np.maximum(x,0)
    x = x.dot(layer_2.data)
    return x

def numpy_eval():
    Y_test_preds_out = predict(X_test.reshape((-1,28*28)))
    Y_test_preds = np.argmax(Y_test_preds_out,axis=1)
    return (Y_test==Y_test_preds).mean()
console.print(f"测试准确度:{numpy_eval()}")