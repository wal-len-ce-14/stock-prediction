import yfinance as yf
from torch.utils.data import Dataset
import torch
import numpy as np
import time
np.set_printoptions(suppress=True, precision=2)


# stock_id = 2344
# tw2618= yf.Ticker(f"{stock_id}.TW")

# d10 = tw2618.history('60m', '15m')

# print(d10['Open'].values)
# ary = d10[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()



class Stocks():
    def __init__(self, stock_id):
        self.stock = yf.Ticker(f"{stock_id}.TW")
        ## 個時段數據
        self.threeyear = self.stock.history('1y', '1d')
        # time.sleep(3)
        # self.quarter = self.stock.history('4mo', '1d')
        # time.sleep(3)
        # self.week = self.stock.history('5d', '15m')
        # time.sleep(3)
        # self.day = self.stock.history('1d', '5m')

    def get_price_year(self):
        return self.threeyear['Close'].values
    
class Data(Dataset):
    def __init__(self, path, train_window):  
        self.path_info = path
        self.train_window = train_window
    def __len__(self):
        return len(self.path_info)-self.train_window
    def __getitem__(self, index):
        info = np.array([])
        for i in range(index, index+self.train_window):
            info = np.append(info, self.path_info[i])
        return info, self.path_info[i]