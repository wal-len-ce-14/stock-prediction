import twstock as tws
import yfinance as yf
import pandas as pd
import numpy as np
import time


# {
#   2330台積電, 2344華邦電, 2303聯電, 2388威盛, 2402毅嘉, 3035智原, 2618長榮航, 2313華通, 
#   3037欣興, 2883開發金, 2882國泰金,
# }
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
            print(f"get stock info from {year} ...")
            time.sleep(1)
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
            'close',
            'capacity', 
            'change',
            'transaction'
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
        if (row['tomorrow_change']) > 0.005: 
            return 1
        else:
            return 0

    def add_target_info(self, close='close', change='change'):
        self.stock['tomorrow_change'] = (self.stock[change].shift(-1) / self.stock[close])  # set target data
        self.stock['target'] = self.stock.apply(self.classification, axis=1)
        self.stock['tomorrow_change'] = self.stock['tomorrow_change'].apply(lambda x: format(x, ".2%"))

    def add_moving_average_info(self):
        horizons = [2,5,20,60, 120, 240]
        new_predictor = []

        for horizon in horizons:
            rolling_avg = (self.stock['close']-self.stock['close'].rolling(horizon).mean()) / self.stock['close']
            rolling_avg_column = f"avg_{horizon}"
            self.stock[rolling_avg_column] = rolling_avg
            trend_column = f"trend_{horizon}"
            # 前幾天的上漲天數總和 因為trend不能包含到今天的資訊 要在shift(1)
            self.stock[trend_column] = self.stock.shift(1).rolling(horizon)["target"].sum() 
            new_predictor += [rolling_avg_column, trend_column]
        self.prodictors += new_predictor
        return self.prodictors

    def add_BBands_info(self):
        horizons = [20, 60, 120, 240]
        new_predictor = []

        for horizon in horizons:
            rolling_std = (self.stock['close'].rolling(horizon)).std(ddof=1)
            std_col = f"std_{horizon}"
            self.stock[std_col] = rolling_std

            mean = np.array(self.stock['close'].rolling(horizon).mean())
            BBand_up = mean + rolling_std*2
            BBand_down = mean - rolling_std*2
            std_up_col = f"BBand_up_{horizon}"
            std_down_col = f"BBand_down_{horizon}"
            self.stock[std_up_col] = BBand_up
            self.stock[std_down_col] = BBand_down

            new_predictor += [std_col, std_up_col, std_down_col]
        self.prodictors += new_predictor
        return self.prodictors

    def Forest_model(self, split, n_estimators=500, min_samples_split=50, random_state=1):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
        train = self.stock.iloc[:-(split)]
        test = self.stock.iloc[-(split):]
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
        precision = precision_score(test["target"], preds, average='micro')
        accuracy = accuracy_score(test["target"], preds)
        print('#########\n')
        print(f'accuracy {accuracy:.4f}')
        return [model, precision, accuracy]

    def drop_Nan(self):
        self.stock = self.stock.dropna() 
    
    def to_test(self):
        print(self.stock.tail(1))
        return self.stock.tail(5)


stock = STOCK(3037, 2022)
stock.add_target_info()

stock.drop_Nan()
stock.add_moving_average_info()
stock.add_BBands_info()
stock.drop_Nan()
# model = stock.Forest_model(split=200, n_estimators=200, min_samples_split=90)

import joblib
# joblib.dump(model, 'model')



loaded_model = joblib.load('model')
print(stock.to_test().iloc[1:])
# result = loaded_model.predict(stock.to_test().iloc[0].values)
# print(result)

# stock.stock = stock.stock.dropna()
# print("###### pri_vol ######")
# stock.add_target_info()
# pc_vol_best = 0
# pc_vol_best_t = []
# for n in range(100, 1000, 100):
#     for min in range(10, 100, 10):
#         acc = stock.Forest_model(split=100,n_estimators=n, min_samples_split=min)[2]
#         if acc > pc_vol_best:
#             pc_vol_best = acc
#             pc_vol_best_t = [n,min]
#         print()

# print("###### avg ######")
# stock.prodictors += stock.add_moving_average_info()
# avg_best = 0
# avg_best_t = []
# for n in range(100, 1000, 100):
#     for min in range(10, 100, 10):
#         acc = stock.Forest_model(split=100,n_estimators=n, min_samples_split=min)[2]
#         if acc > avg_best:
#             avg_best = acc
#             avg_best_t = [n,min]
#         print()
# print("######################\n")
# print('pc_vol_best', pc_vol_best)
# print('pc_vol_best_t', pc_vol_best_t)
# print('avg_best', avg_best)
# print('avg_best_t', avg_best_t)







