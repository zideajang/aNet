import os
import hashlib

import gzip
import numpy as np
from tqdm import trange

import matplotlib.pyplot as plt
from rich.console import Console

from anet import Tensor
from anet.core.optim import Optimizer,SGD,Adam

console = Console()

def mnist():

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
    return X_train, Y_train, X_test, Y_test
X_train, Y_train, X_test, Y_test = mnist()




# 定义模型
def layer_init(input_size,hidden_size):
    # 参数初始 (-1/sqrt(in*out),1/sqrt(in*out))
    ret = np.random.uniform(-1.,1.,size=(input_size,hidden_size))/np.sqrt(input_size*hidden_size)
    return ret.astype(np.float32)

class Net:
    def __init__(self):
        self.layer_1 = Tensor(layer_init(784,128))
        self.layer_2 = Tensor(layer_init(128,10))

    def forward(self,x):
        # 全连接层
        x = x.dot(self.layer_1)
        # ReLU 激活层
        x = x.relu()
        # 全连接
        x = x.dot(self.layer_2)
        # logic 层
        x = x.logsoftmax()
        return x

model = Net()
optim = Adam([model.layer_1,model.layer_2],lr=0.01)

# 训练
# 超参数
batch_size= 128
learning_rate = 0.01
epoch = 1000

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

    outs = model.forward(x)

    
    # xy
    # NLL Loss
    loss = outs.mul(y).mean()
    # 计算loss
    # 反向传播
    loss.backward()
    # 用优化器来更新参数
    optim.step()

    cat = np.argmax(outs.data,axis=1)
    acc = (cat == Y).mean()


    losses.append(loss)
    accuracies.append(acc)
    t.set_description(f"loss {loss} accuracy:{acc}")



def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1,28*28))))
    Y_test_preds = np.argmax(Y_test_preds_out.data,axis=1)
    return (Y_test==Y_test_preds).mean()

console.print(f"测试准确度:{numpy_eval()}")

# Create a figure with two subplots
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot([loss.data for loss in losses], label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
plt.plot(accuracies, label='Training Accuracy', color='orange')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Display the plots
plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.show()