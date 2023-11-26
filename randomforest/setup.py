import twstock as tws
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime



# tws.__update_codes()

def get_stock_history(stockid, start_year, end_year):
        if start_year >= end_year+1:
            print("[-] start_year >= end_year!!")
            return -1
        colume = [
            'date',
            'close',
            'capacity',
            'change',
            'transaction'
        ]
        info = [[]]
        stock = tws.Stock(f"{stockid}")
        for year in range(start_year, end_year+1):
            print(f"[+] get stock info from {year} ...")
            time.sleep(0.2)
            for month in range(1, 13):
                stock_infos = stock.fetch(year,month)
                for idx, stock_info in enumerate(stock_infos):
                    new_info = np.array([[  
                        stock_info.date.strftime("%Y-%m-%d"),
                        stock_info.close,
                        stock_info.capacity, 
                        stock_info.change,
                        stock_info.transaction]])
                    if month == 1 and year == start_year and idx == 0:
                        info = new_info
                    else:
                        info = np.append(info, new_info, axis=0)
        # print(f"get stock info from {year+1} ...")
        stockpd = pd.DataFrame(          # get stock data
            info, 
            columns=colume,
        )
        stockpd[['close',
            'capacity',
            'change',
            'transaction'
        ]] = stockpd[['close',
            'capacity',
            'change',
            'transaction'
        ]].astype('float64')
        stockpd = stockpd.set_index('date')
        return stockpd

class STOCK():
    def __init__(self, stockid, start_year, end_year=2023):
        self.stockid = stockid
        self.start = start_year
        self.end = end_year
        self.stock = get_stock_history(stockid, start_year, end_year)
        self.prodictors = [
            # 'close',
            # 'capacity', 
            # 'change',
            # 'transaction'
        ]

    def classification(self, row):
        # if (row['tomorrow_change']) > 0.065:
        #     return 3
        # if (row['tomorrow_change']) > 0.03:
        #     return 2
        # elif (row['tomorrow_change']) > 0.007:
        #     return 1
        # elif (row['tomorrow_change']) < -0.07:
        #     return -3
        # elif (row['tomorrow_change']) < -0.04:
        #     return -2
        # elif (row['tomorrow_change']) < -0.07:
        #     return -1
        # else:
        #     return 0
        if (row['tomorrow_change']) > 0: 
            return 1
        else:
            return 0

    def add_target_info(self, close='close', change='change'):  # 一般資訊
        self.stock['change_ratio'] = (self.stock[change] / self.stock[close].shift(1))*100
        self.stock['tomorrow_change'] = (self.stock[change].shift(-1) / self.stock[close])  # set target data
        self.stock['target'] = self.stock.apply(self.classification, axis=1)
        self.stock['tomorrow_change'] = self.stock['tomorrow_change'].apply(lambda x: format(x, ".2%"))
        self.prodictors += ['change_ratio']
        
    def add_moving_average_info(self):  # 均線比率
        horizons = [2,5,20,60, 120, 240]
        new_predictor = []

        for horizon in horizons:
            rolling_avg = ((self.stock['close']-self.stock['close'].rolling(horizon).mean()) / self.stock['close'])*100
            rolling_avg_column = f"avg_{horizon}"
            self.stock[rolling_avg_column] = rolling_avg
            trend_column = f"trend_{horizon}"
            # 前幾天的上漲天數總和 因為trend不能包含到今天的資訊 要在shift(1)
            self.stock[trend_column] = self.stock.shift(1).rolling(horizon)["target"].sum() 
            new_predictor += [rolling_avg_column, trend_column]
            # 量平均
            rolling_avg = (self.stock['capacity']-self.stock['capacity'].rolling(horizon).mean()) / self.stock['capacity']
            rolling_avg_column = f"cap_avg_{horizon}"
            self.stock[rolling_avg_column] = rolling_avg*100
            new_predictor += [rolling_avg_column]
            rolling_avg = (self.stock['transaction']-self.stock['transaction'].rolling(horizon).mean()) / self.stock['transaction']
            rolling_avg_column = f"'trans_avg_{horizon}"
            self.stock[rolling_avg_column] = rolling_avg*100
            new_predictor += [rolling_avg_column]

        self.prodictors += new_predictor
        return self.prodictors

    def add_BBands_info(self):  # 布林通道
        horizons = [5, 10, 20, 60, 90, 120, 240]
        new_predictor = []

        for horizon in horizons:
            rolling_std = (self.stock['close'].rolling(horizon)).std(ddof=1)
            std_col = f"std_{horizon}"
            self.stock[std_col] = rolling_std

            mean = np.array(self.stock['close'].rolling(horizon).mean())
            BBand_up = ((mean + rolling_std*2)-self.stock['close']) / self.stock['close']
            BBand_down = ((mean - rolling_std*2)-self.stock['close']) / self.stock['close']
            std_up_col = f"BBand_up_{horizon}"
            std_down_col = f"BBand_down_{horizon}"
            self.stock[std_up_col] = BBand_up*100
            self.stock[std_down_col] = BBand_down*100

            new_predictor += [std_up_col, std_down_col]
        self.prodictors += new_predictor
        return self.prodictors

    def add_Leverage(self): # 籌碼
        from scraping import get_juridical_person as get
        Leverage = get(self.stockid, self.start)
        self.stock = pd.merge(self.stock, Leverage, on='date', how='inner')
        horizons_sum = [1, 5, 10, 20, 60]
        horizon_ratio = [5, 10, 20, 60, 120]
        new_predictor = []
        lever_col = []
        for horizon in horizons_sum:
            for lever in ['Foreign_Investor','Investment_Trust','Dealer_self','Dealer_Hedging']:
                rolling_sum = self.stock[lever].rolling(horizon).sum()
                leverage_col = f'{lever}_{horizon}'
                lever_col += [leverage_col]
                self.stock[leverage_col] = rolling_sum
        for horizon in horizon_ratio:
            for lever_sum in lever_col:
                rolling_std = (self.stock[lever_sum].rolling(horizon)).std(ddof=1)
                up_std = self.stock[lever_sum]+2*rolling_std
                down_std = self.stock[lever_sum]-2*rolling_std
                mean = (self.stock[lever_sum].rolling(horizon)).mean()
                Leverage_bias_up = f'{lever_sum}_up_{horizon}'
                Leverage_bias_down = f'{lever_sum}_down_{horizon}'
                Leverage_mean = f'{lever_sum}_mean_{horizon}'
                up_ratio = up_std / self.stock[lever_sum]
                down_ratio = down_std / self.stock[lever_sum]
                mean_ratio = mean / self.stock[lever_sum]


                self.stock[Leverage_bias_up] = up_ratio*100
                self.stock[Leverage_bias_down] = down_ratio*100
                self.stock[Leverage_mean] = mean_ratio*100

                self.stock.loc[self.stock[Leverage_bias_up] > 1000000] = 1000000
                self.stock.loc[self.stock[Leverage_bias_down] > 1000000] = 1000000
                self.stock.loc[self.stock[Leverage_mean] > 1000000] = 1000000

                new_predictor += [Leverage_bias_up, Leverage_bias_down, Leverage_mean]

        self.prodictors += new_predictor
        
    def Forest_model(self, split, n_estimators=800, min_samples_split=70, random_state=1, val=50):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
        real_world = self.stock.iloc[:-(val)]
        train = real_world.iloc[:-(split)]
        test = real_world.iloc[-(split):]
        model.fit(train[self.prodictors], train["target"])
        # 
        from sklearn.metrics import precision_score, accuracy_score
        preds = model.predict(test[self.prodictors])
        preds = pd.Series(preds)
        precision = precision_score(preds, test["target"], average='macro')
        accuracy = accuracy_score(preds, test["target"])
        print('#########')
        print(f'[+] precision {precision:.4f}\n[+] accuracy {accuracy:.4f}, \n')
        return {
            'model': model,
            'precision': precision, 
            'accuracy': accuracy
        }

    def drop_Nan(self):
        self.stock = self.stock.dropna()
        self.stock = self.stock.round(3) 
        
    
    def to_test(self, val=10):
        # print(self.real_world)
        return self.stock[-(val):]

# stock = STOCK(2344, 2020,2023)
# stock.add_Leverage()
# stock.add_target_info()
# stock.add_moving_average_info()
# stock.add_BBands_info()
# stock.add_Leverage()
# print(stock.prodictors)
# stock.drop_Nan()
# model = stock.Forest_model(
#     split=200, 
#     n_estimators=200, 
#     min_samples_split=90
# )








