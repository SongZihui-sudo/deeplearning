import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x: float, w: float, b: float):
    return x * w + b

def squared_loss(y_pred: float, y_true: float):
    return np.power(y_pred - y_true, 2) / 2

def gradient_w(y_hat, y_true, x):
    return (y_hat - y_true) * x

def gradient_b(y_hat, y_true, x):
    return (y_hat - y_true)

def gradient(args: np.ndarray, n: int, train_x: np.ndarray, train_y: np.ndarray, y_hat: np.ndarray, lr: float, grad_func: list):
    res:np.ndarray = np.ones(2)
    j: int = 0
    for arg in args:
        temp: float = 0
        for i in range(0, train_x.size):
            cur = grad_func[j](y_hat[i], train_y[i], train_x[i])
            temp += cur
        arg = arg - (lr * temp) / n
        res[j] = arg
        j = j + 1
    return res

data_set: pd.DataFrame = pd.read_csv("Salary_dataset.csv")

data_set.shape

x : pd.Series = data_set["YearsExperience"]
y : pd.Series = data_set['Salary']
train_set: pd.DataFrame = data_set.sample(frac=0.6, random_state=0, axis=0)
test_set: pd.DataFrame = data_set[~data_set.index.isin(train_set.index)]
x_train: np.ndarray = train_set["YearsExperience"].to_numpy()
y_train: np.ndarray = train_set["Salary"].to_numpy()
x_test: np.ndarray = test_set["YearsExperience"].to_numpy()
y_test: np.ndarray = test_set["Salary"].to_numpy()

plt.scatter(x, y)
plt.show()

w_b: np.ndarray = np.array([1, 0])
y_hat = np.ndarray = np.zeros(100)

for i in range(0, 50):
    j:int = 0
    for c_x in x_train:
        y_hat[j] = f(c_x, w_b[0], w_b[1])
        j = j + 1
    w_b = gradient(w_b, x_train.size, x_train, y_train, y_hat, 0.01, [gradient_w, gradient_b])
    print("第 %d 次迭代 w = %f  b = %f" % (i, w_b[0], w_b[1]))

plt.scatter(x_train, y_train)
plt.show()