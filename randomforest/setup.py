import twstock as tws
import yfinance as yf
import pandas as pd
import numpy as np
import time


# {
#   2344華邦電, 2303聯電, 2388威盛, 2402毅嘉, 3035智原, 2618長榮航, 
# }
# tws.__update_codes()

def get_stock_history(stockid, start_year, end_year):
    colume = [
        'capacity', 
        'open',
        'high',
        'low',
        'close',
        'change',
        'transaction'
    ]
    info = [[]]
    stock = tws.Stock(f"{stockid}")
    for year in range(start_year, end_year):
        print(f"get stock info from {year} ...")
        time.sleep(0.2)
        for month in range(1, 13):
            stock_infos = stock.fetch(year,month)
            for stock_info in stock_infos:
                new_info = np.array([[    
                    stock_info.capacity, 
                    stock_info.open,
                    stock_info.high,
                    stock_info.low,
                    stock_info.close,
                    stock_info.change,
                    stock_info.transaction]])
                if month == 1 and year == start_year:
                    info = new_info
                else:
                    info = np.append(info, new_info, axis=0)
    stockpd = pd.DataFrame(          # get stock data
        info, 
        columns=colume,
        dtype=float
    )
    return stockpd

def classification(row):
    if (row['tomorrow_change']) > 0.065:
        return 3
    elif (row['tomorrow_change']) > 0.03:
        return 2
    elif (row['tomorrow_change']) > 0.007:
        return 1
    elif (row['tomorrow_change']) < -0.07:
        return -3
    elif (row['tomorrow_change']) < -0.04:
        return -2
    elif (row['tomorrow_change']) < -0.07:
        return -1
    else:
        return 0

stock = get_stock_history(2344, 2022, 2023)

stock['tomorrow'] = stock['close'].shift(-1)  # set target data
stock['tomorrow_change'] = (stock['change'].shift(-1) / stock['close'])  # set target data
stock['target'] = stock.apply(classification, axis=1)
stock['tomorrow_change'] = stock['tomorrow_change'].apply(lambda x: format(x, ".2%"))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = stock.iloc[:-100]
test = stock.iloc[-100:]
prodictors = [
    'capacity', 
    'open',
    'high',
    'low',
    'close',
    'change',
    'transaction'
]
model.fit(train[prodictors], train["target"])

from sklearn.metrics import precision_score
preds = model.predict(test[prodictors])
preds = pd.Series(preds, test["target"])
print('preds\n', preds)
precision = precision_score(test["target"], preds, average='micro')
print('precision', precision)
print('preds\n', preds)