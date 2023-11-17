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


# stock_id = 2344
# tw2618= yf.Ticker(f"{stock_id}.TW")

# d10 = tw2618.history('60m', '15m')

# print(d10['Open'].values)
# ary = d10[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()



class Stocks():
    def __init__(self, stock_id):
        self.data = np.array([])
        ss = Stock(f"{stock_id}")
        for i in range(1,13):
            mess = ss.fetch(2020,i)
            for st in mess:
                self.data = np.append(self.data, st.close)
        print("wait...")
        for i in range(1,13):
            mess = ss.fetch(2021,i)
            for st in mess:
                self.data = np.append(self.data, st.close)
        print("wait...")
        for i in range(1,13):
            mess = ss.fetch(2022,i)
            for st in mess:
                self.data = np.append(self.data, st.close)
        print("wait...")
        time.sleep(3)
        for i in range(1,13):
            mess = mess = ss.fetch(2023,i)
            for st in mess:
                self.data = np.append(self.data, st.close)
                
        # self.stock = Stock(f"{stock_id}")
        # self.stock = yf.Ticker(f"{stock_id}.TW")
        ## 個時段數據
        # self.year = self.stock.price()
        # print(self.year)
        # self.year = self.stock.history('1y', '1d')
        # time.sleep(3)
        # self.recent = self.stock.history('3mo', '1d')

        # print(np.round(self.year['Close'].values,2))
    def get_price_year(self):
        return self.data
    def get_recent(self):
        return self.data[-10:]
    def get_testing_date(self, period):
        return torch.tensor(self.data[-period:-1]).to(torch.float32), torch.tensor(self.data[-1]).to(torch.float32)
            # def get_price_year(self):
            #     return self.year['Close'].values
            # def get_recent(self):
            #     # print(self.recent['Close'].tail(10))
            #     return self.recent
    # def get_testing_date(self, start, period):
    #     startdate = datetime.strptime(start, "%Y-%m-%d")
    #     testing_date = self.stock.history(start=start, end=startdate-timedelta(days=period))['Close'].values
    #     return testing_date[:-1], testing_date[-1]

    
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
    
