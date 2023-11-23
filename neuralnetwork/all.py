import yfinance as yf
import twstock
import numpy as np
import torch 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime


# ss = twstock.Stock('2303')
# print((ss.fetch(2023,3)))


# aa = []
# aa = np.append(aa, [[]],axis=1)

# aa = [1,2]
# print(aa)
# print(np.append(aa, [[9,8,7,6,5,4,3,2,1]], axis=0))
# print(np.append(aa, [[8,7]]))

# aa = torch.randn(900,3)
# print(aa.shape)
# aa = torch.nn.Flatten(0,1)(aa)
# print(aa)
# aa = torch.flatten(aa)
# print(aa)

# data = np.array([[1,20,1],[1,30,1],[1,80,1],[1,110,1],[1,50,1],[1,10,1],[1,40,1],[1,30,1]])

# mean = np.mean(data[:,1])
# std_value = np.std(data[:,1])
# normalized = (data[:,1] - mean) / std_value
# data[:,1] = normalized
# print(mean, std_value, normalized)

# import numpy as np
# import pandas as pd

# # 假设 self.stock 是一个 Pandas DataFrame，包含 'close' 列
# horizon = 20
# stock = pd.DataFrame(          # get stock data
#             [[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6],[7,8,9],[1,2,3]], 
#             columns=['idx', 'close', 'num'],
#             dtype=float
#         )
# # 计算滚动窗口均值和标准差
# rolling_mean = stock.rolling(3).mean()
# rolling_std = stock.rolling(3).std()

# # 使用 numpy 进行计算，确保形状兼容
# BBand_up = rolling_mean + 2 * np.array(rolling_std)
# print(stock.rolling(3).std(), stock.rolling(3).mean())
# print(stock)

all = {}

new ={
    'tt' : {'aa': 123}
}

all.update(new)

new ={
    'bb' : {'aa': 123}
}
all.update(new)
print(all)