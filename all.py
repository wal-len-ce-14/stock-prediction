import yfinance as yf
import twstock
import numpy as np
import torch 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime


ss = twstock.Stock('2303')
print((ss.fetch(2023,3)))


# aa = np.empty((1,2))
# aa = np.append(aa, [[]],axis=1)

# print(aa)
# print(np.append(aa, [[8,7]], axis=0))
# print((np.append(aa, [[8,7]], axis=0))[0])

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

