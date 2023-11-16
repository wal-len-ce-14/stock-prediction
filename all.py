import yfinance as yf
import numpy as np
import torch 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

info = np.array([[0],[1,2],[0,[1,2]]])
print(info)
info = np.append(info, 1)
print(info)