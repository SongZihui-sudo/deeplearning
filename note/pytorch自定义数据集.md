# 自定义数据集

创建 datasets ，也就是所需要读取的数据集。  
把 datasets 传入DataLoader。  
DataLoader迭代产生训练数据提供给模型。  

[链接](https://blog.csdn.net/qq_40788447/article/details/114937779)  

一个Map式的数据集必须要重写 __getitem__(self, index)、 __len__(self) 两个方法，用来表示从索引到样本的映射(Map)。 __getitem__(self, index)按索引映射到对应的数据， __len__(self)则会返回这个数据集的长度。  

```python
import torch.utils.data as data

class VOCDetection(data.Dataset):
    '''
    必须继承data.Dataset类
    '''

    def __init__(self):
        '''
        在这里进行初始化，一般是初始化文件路径或文件列表
        '''
        pass

    def __getitem__(self, index):
        '''
        1. 按照index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）
        2. 对读取到的数据进行数据增强 (数据增强是深度学习中经常用到的，可以提高模型的泛化能力)
        3. 返回数据对 （一般我们要返回 图片，对应的标签） 在这里因为我没有写完整的代码，返回值用 0 代替
        '''
        return 0

    def __len__(self):
        '''
        返回数据集的长度
        '''
        return 0
```