import twstock as tws
import yfinance as yf
import pandas as pd
import numpy as np
import time



# tws.__update_codes()

def get_stock_history(stockid, start_year, end_year):
        if start_year >= end_year+1:
            print("[-] start_year >= end_year!!")
            return -1
        colume = [
            'close',
            'capacity',
            'change',
            'transaction'
        ]
        info = [[]]
        stock = tws.Stock(f"{stockid}")
        for year in range(start_year, end_year+1):
            print(f"[+] get stock info from {year} ...")
            time.sleep(0.1)
            for month in range(1, 13):
                stock_infos = stock.fetch(year,month)
                for stock_info in stock_infos:
                    new_info = np.array([[  
                        stock_info.close,
                        stock_info.capacity, 
                        stock_info.change,
                        stock_info.transaction]])
                    if month == 1 and year == start_year:
                        info = new_info
                    else:
                        info = np.append(info, new_info, axis=0)
        # print(f"get stock info from {year+1} ...")
        stockpd = pd.DataFrame(          # get stock data
            info, 
            columns=colume,
            dtype=float
        )
        return stockpd

class STOCK():
    def __init__(self, stockid, start_year, end_year=2023):
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

    def add_target_info(self, close='close', change='change'):
        self.stock['change_ratio'] = (self.stock[change] / self.stock[close].shift(1))
        self.stock['tomorrow_change'] = (self.stock[change].shift(-1) / self.stock[close])  # set target data
        self.stock['target'] = self.stock.apply(self.classification, axis=1)
        self.stock['tomorrow_change'] = self.stock['tomorrow_change'].apply(lambda x: format(x, ".2%"))
        self.prodictors += ['change_ratio']
        
    def add_moving_average_info(self):
        horizons = [2,5,20,60, 120, 240, 500]
        new_predictor = []

        for horizon in horizons:
            rolling_avg = (self.stock['close']-self.stock['close'].rolling(horizon).mean()) / self.stock['close']
            rolling_avg_column = f"avg_{horizon}"
            self.stock[rolling_avg_column] = rolling_avg
            trend_column = f"trend_{horizon}"
            # 前幾天的上漲天數總和 因為trend不能包含到今天的資訊 要在shift(1)
            self.stock[trend_column] = self.stock.shift(1).rolling(horizon)["target"].sum() 
            new_predictor += [rolling_avg_column, trend_column]
            # 量平均
            rolling_avg = (self.stock['capacity']-self.stock['capacity'].rolling(horizon).mean()) / self.stock['capacity']
            rolling_avg_column = f"cap_avg_{horizon}"
            self.stock[rolling_avg_column] = rolling_avg
            new_predictor += [rolling_avg_column]
            rolling_avg = (self.stock['transaction']-self.stock['transaction'].rolling(horizon).mean()) / self.stock['transaction']
            rolling_avg_column = f"'trans_avg_{horizon}"
            self.stock[rolling_avg_column] = rolling_avg
            new_predictor += [rolling_avg_column]

        self.prodictors += new_predictor
        return self.prodictors

    def add_BBands_info(self):
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
            self.stock[std_up_col] = BBand_up
            self.stock[std_down_col] = BBand_down

            new_predictor += [std_up_col, std_down_col]
        self.prodictors += new_predictor
        return self.prodictors

    def Forest_model(self, split, n_estimators=500, min_samples_split=50, random_state=1, val=50):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
        real_world = self.stock.iloc[:-(val)]
        train = real_world.iloc[:-(split)]
        test = real_world.iloc[-(split):]
        model.fit(train[self.prodictors], train["target"])
        # 
        from sklearn.metrics import precision_score, accuracy_score

        # use train data to test
            # preds = model.predict(train[self.prodictors].iloc[-20:])
            # print("train target", train["target"].iloc[-20:].values, '\n#########')
            # print('train preds', preds)

        # use test data to test
            # preds_proba = model.predict_proba(test[self.prodictors])
            # preds[preds >= 0.6] = 1
            # preds[preds < 0.6] = 0
        preds = model.predict(test[self.prodictors])
        preds = pd.Series(preds)
        precision = precision_score(test["target"].values, preds, average='micro')
        accuracy = accuracy_score(test["target"].values, preds)
        print('#########')
        print(f'[+] precision {precision:.4f}\n[+] accuracy {accuracy:.4f}, \n')
        return {
            'model': model,
            'precision': precision, 
            'accuracy': accuracy
        }

    def drop_Nan(self):
        self.stock = self.stock.dropna() 
    
    def to_test(self, val=10):
        # print(self.real_world)
        return self.stock[-(val):]

# stock = STOCK(3037, 2022)
# stock.add_target_info()
# stock.add_moving_average_info()
# stock.add_BBands_info()
# stock.drop_Nan()
# model = stock.Forest_model(
#     split=200, 
#     n_estimators=200, 
#     min_samples_split=90
# )








