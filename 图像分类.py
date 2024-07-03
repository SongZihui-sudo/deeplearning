import pandas
import dataset.fashion_mnist.utils.mnist_reader as mnist_reader
import torch
import tqdm
import sklearn.metrics as sm
import numpy as np

X_train, y_train = mnist_reader.load_mnist('dataset/fashion_mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('dataset/fashion_mnist/data/fashion', kind='t10k')

# 使用 GPU
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# 多层感知机
class MLP(torch.nn.Module):
    def __init__(self,num_i, num_h1, num_o):
        super(MLP,self).__init__()
        self.input_layout = torch.nn.Linear(num_i, num_h1)
        self.relu1 = torch.nn.ReLU()
        self.hide1_layout = torch.nn.Linear(num_h1, num_h1)
        self.relu2 = torch.nn.ReLU()
        self.output_layout = torch.nn.Linear(num_h1, num_o)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.input_layout(x)
        x = self.relu1(x)
        x = self.hide1_layout(x)
        x = self.relu2(x)
        x = self.output_layout(x)
        return x

x_train_tensor = torch.tensor(X_train, requires_grad=True, dtype=torch.float, device=try_gpu())
y_train_tensor = torch.tensor(y_train, requires_grad=True, dtype=torch.float, device=try_gpu())
x_test_tensor = torch.tensor(X_test, dtype=torch.float, device=try_gpu())
y_test_tensor = torch.tensor(y_test, dtype=torch.float, device=try_gpu())

# 训练
acc_max = 0
arg_max = 0

model = MLP(x_train_tensor.shape[1], 256, 10)
model = model.to(device=try_gpu())

cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
for i in tqdm.tqdm(range(5000)):
    y_predict = model(x_train_tensor)
    y_predict = torch.argmax(y_predict, dim=1)
    y_predict = y_predict.type(torch.float)
    optimizer.zero_grad()
    loss = cost(y_train_tensor, y_predict.reshape(-1) )
    loss.backward()
    optimizer.step()

y_predict = model(x_test_tensor)
y_predict = torch.argmax(y_predict, dim=1)
y_predict = y_predict.type(torch.float)

# 计算准确率
y_predict = torch.round(y_predict)
accuracy = torch.eq(y_predict.squeeze(dim=-1), y_test_tensor).float().mean()

print("accuracy: %f %" % (accuracy * 100))