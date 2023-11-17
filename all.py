import yfinance as yf
import twstock
import numpy as np
import torch 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime


ss = twstock.Stock('2303')
print(len(ss.fetch(2023,3)))


# aa = np.array([1,2,3])
# print(len(aa))

