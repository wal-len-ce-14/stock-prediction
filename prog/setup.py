import twstock as tws
# import yfinance as yf
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
            'transaction'
        ]
        info = [[]]
        stock = tws.Stock(f"{stockid}")
        for year in range(start_year, end_year+1):
            print(f"\t[+] get stock info from {year} ...")
            time.sleep(0.2)
            for month in range(1, 13):
                stock_infos = stock.fetch(year,month)
                for idx, stock_info in enumerate(stock_infos):
                    new_info = np.array([[  
                        stock_info.date.strftime("%Y-%m-%d"),
                        stock_info.close,
                        stock_info.capacity, 
                        stock_info.transaction]])
                    if month == 1 and year == start_year and idx == 0:
                        info = new_info
                    else:
                        info = np.append(info, new_info, axis=0)
        stockpd = pd.DataFrame(          # get stock data
            info, 
            columns=colume,
        )
        stockpd[['close',
            'capacity',
            'transaction'
        ]] = stockpd[['close',
            'capacity',
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
        self.preserve = self.stock.tail(1)
        self.prodictors = []

    def classification(self, row, day=1):
        # if (row[f'change_value{day}']) > 0.2: 
        #     return 4
        # elif (row[f'change_value{day}']) > 0.07: 
        #     return 3
        # elif (row[f'change_value{day}']) > 0.04: 
        #     return 2
        # elif (row[f'change_value{day}']) > 0.005: 
        #     return 1
        # elif (row[f'change_value{day}']) < -0.1: 
        #     return -3
        # elif (row[f'change_value{day}']) < -0.05: 
        #     return -2
        # elif (row[f'change_value{day}']) < -0.02: 
        #     return -1
        if (row[f'change_value{day}']) > 0.001:
            return 1
        else:
            return 0

    def add_target_info(self, close='close'):  # 一般資訊
        self.stock['change_value1'] = ((self.stock[close] - self.stock[close].shift(1)) / self.stock[close].shift(1))
        self.stock['change_value2'] = ((self.stock[close] - self.stock[close].shift(2)) / self.stock[close].shift(2))
        self.stock['change_value5'] = ((self.stock[close] - self.stock[close].shift(5)) / self.stock[close].shift(5))
        self.stock['change_value10'] = ((self.stock[close] - self.stock[close].shift(10)) / self.stock[close].shift(10))

        self.stock['target'] = self.stock.apply(self.classification, axis=1, args=(1,))
        self.stock['target1'] = self.stock.apply(self.classification, axis=1, args=(1,))
        self.stock['target2'] = self.stock.apply(self.classification, axis=1, args=(2,))
        self.stock['target5'] = self.stock.apply(self.classification, axis=1, args=(5,))
        self.stock['target10'] = self.stock.apply(self.classification, axis=1, args=(10,))

        self.stock['short_term'] = (self.stock['change_value5'] + self.stock['change_value2']) / 2
        self.stock['mid_term'] = (self.stock['change_value10'] - self.stock['change_value5']) / 2

        self.prodictors += ['change_value1', 'change_value2', 'change_value5', 'change_value10', 'short_term', 'mid_term']
        
    def add_moving_average_info(self):  # 均線比率
        horizons = [2,5,10,20,60]
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
        horizons = [2,5,10,20,60]
        new_predictor = []

        for horizon in horizons:
            rolling_std = (self.stock['close'].rolling(horizon)).std(ddof=1)
            std_col = f"std_{horizon}"
            self.stock[std_col] = rolling_std

            mean = np.array(self.stock['close'].rolling(horizon).mean())
            BBand_up = ((mean + rolling_std*1.8)-self.stock['close']) / self.stock['close']
            BBand_down = ((mean - rolling_std*1.8)-self.stock['close']) / self.stock['close']
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
        horizons_sum = [1,3,5,10,20,40,60,120]
        new_predictor = []
        for horizon in horizons_sum:
            for lever in ['Foreign_Investor','Investment_Trust','Dealer_self','Dealer_Hedging']:
                interval_ratio = self.stock[lever] / self.stock[lever].rolling(horizon).sum().replace(0,1)
                interval_ratio_col = f'{lever}_{horizon}'
                self.stock[interval_ratio_col] = interval_ratio
                new_predictor += [interval_ratio_col]
        self.prodictors += new_predictor

    def add_Margin(self):
        from scraping import get_Margin_info as get
        margin = get(self.stockid, self.start)
        self.stock = pd.merge(self.stock, margin, on='date', how='inner')
        self.prodictors += ['Margin', 'Sale']
        horizons = [2,3,5,10,20,60]
        new_predictor = []
        for horizon in horizons:
            avg_M = self.stock['Margin'].rolling(horizon).mean()
            avg_col_M = f'Margin_{horizon}_avg'
            self.stock[avg_col_M] = avg_M
            new_predictor += [avg_col_M]

            avgS = self.stock['Sale'].rolling(horizon).mean()
            avg_colS = f'Sale_{horizon}_avg'
            self.stock[avg_colS] = avgS
            new_predictor += [avg_colS]

        self.prodictors += new_predictor
  
    def Forest_model(self, split, n_estimators=800, min_samples_split=70, random_state=1, depth=6, val=25, target_day=1):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=depth, random_state=random_state)
        real_world = self.stock.iloc[:-(val)]
        train = real_world.iloc[:-(split)]
        test = real_world.iloc[-(split):]
        model.fit(train[self.prodictors], train[f"target{target_day}"])
        # 
        from sklearn.metrics import precision_score, accuracy_score
        preds = model.predict(test[self.prodictors])
        preds = pd.Series(preds)
        precision = precision_score(preds.values, test[f"target{target_day}"].values, average='macro')
        accuracy = accuracy_score(preds.values, test[f"target{target_day}"].values)
        print('######### train test #########')
        print(f'[+] precision {precision:.4f}\n[+] accuracy {accuracy:.4f}, \n')
        return {
            'model': model,
            'precision': precision, 
            'accuracy': accuracy
        }

    def drop_Nan(self):
        self.preserve = self.stock
        self.stock = self.stock.dropna()
        self.stock = self.stock.round(3) 
          
    def to_test(self, val=10):  # 回傳從未train過的test資料
        # print(self.real_world)
        return self.stock[-(val):] 

    def predict_tomorrow(self):
        return self.preserve.index[-1]
    def get(self, bias=0):
        if bias == 0:
            return self.preserve.loc[self.preserve.index[-len(self.prodictors):],self.prodictors], self.preserve.index[-1], self.preserve.loc[self.preserve.index[-1], 'target5']
        elif bias > 0:
            return self.preserve.loc[self.preserve.index[-len(self.prodictors)-bias:-bias],self.prodictors], self.preserve.index[-1-bias], self.preserve.loc[self.preserve.index[-1-bias], 'target5']

# stock = STOCK(2330, 2023,2023)

# stock.add_target_info()
# stock.add_moving_average_info()
# stock.add_BBands_info()
# stock.add_Leverage()
# stock.add_Margin()
# stock.drop_Nan()
# print(stock.get(bias=1))
# print(stock.stock['change_value5'][:20].shape[0])
# print(stock.stock['change_value5'][:20].values.shape[0])
# print(stock.stock['change_value5'][:20].values)
# print(stock.stock[['change_value1', 'change_value2', 'change_value5', 'change_value10']].shape[1])
# model = stock.Forest_model(
#     split=200, 
#     n_estimators=200, 
#     min_samples_split=90
# )








