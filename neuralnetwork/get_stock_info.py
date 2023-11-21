import yfinance as yf
import twstock
from twstock import Stock
from torch.utils.data import Dataset
import torch
import numpy as np
import time
from datetime import datetime, timedelta
# twstock.__update_codes()

np.set_printoptions(suppress=True, precision=2)

def init_data(stock):
    for i in range(1,13):
        mess =stock.fetch(2020,i)
        for idx, st in enumerate(mess):
            if i == 1 and idx == 0:
                data = np.array([[st.close, st.capacity, st.change]])
            else:
                data = np.append(data, [[st.close, st.capacity, st.change]], axis=0)
    print("wait...")
    for i in range(1,13):
        mess = stock.fetch(2021,i)
        for st in mess:
            data = np.append(data, [[st.close, st.capacity, st.change]], axis=0)
    print("wait...")
    for i in range(1,13):
        mess = stock.fetch(2022,i)
        for st in mess:
            data = np.append(data, [[st.close, st.capacity, st.change]], axis=0)
    print("wait...")
    time.sleep(3)
    for i in range(1,13):
        mess = mess = stock.fetch(2023,i)
        for st in mess:
            data = np.append(data, [[st.close, st.capacity, st.change]], axis=0)
    for i in range(1,2):
        mean = np.mean(data[:,i])
        std_value = np.std(data[:,i])
        normalized = (data[:,i] - mean) / std_value
        data[:,i] = normalized
    # print(mean, std_value, normalized)

    return data

class Stocks():
    def __init__(self, stock_id):
        self.stock = Stock(f"{stock_id}")
        self.data = init_data(self.stock)

    def get_price_year(self):
        return self.data
    def get_recent(self):
        return self.data[-10:,0]
    def get_testing_date(self, period):
        return torch.tensor(self.data[-(period):-1]).to(torch.float32).unsqueeze(0), torch.tensor(self.data[-1][0]).to(torch.float32)

s = Stocks(2344)
print(type(s))
    
class Data(Dataset):
    def __init__(self, path, train_window):  
        self.path_info = path
        self.train_window = train_window
        
    def __len__(self):
        
        return len(self.path_info)-self.train_window
    def __getitem__(self, index):
        for i in range(index, index+self.train_window):
            if i == index: 
                info = np.array([self.path_info[i]])
            else:
                info = np.append(info, [self.path_info[i]], axis=0)

        return info, self.path_info[i][0]
    
