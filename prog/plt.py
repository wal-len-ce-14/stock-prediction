from scraping import get_price_detail as get
import pandas as pd
from scraping import get_juridical_person as corp

stockID = 3032


stock = get(stockID, 2023, 2024)
stock['vol_status'] = (round(((stock['close']*(stock['capacity'])).rolling(7).sum()*(stock['capacity']))/((stock['capacity']**2).rolling(7).sum()), 2))/stock['close'] 
stock['avg10'] = round(((stock['close']*(stock['capacity'])).rolling(10).sum())/(stock['capacity']).rolling(10).sum(), 2)
stock['avg5'] = round(((stock['close']*(stock['capacity'])).rolling(5).sum())/(stock['capacity']).rolling(5).sum(), 2)
stock['deviations5'] = (stock['close']-stock['avg5']) / stock['avg5']
stock['deviations10'] = (stock['close']-stock['avg10']) / stock['avg10']
Leverage = corp(stockID,2023, 2024)
stock["Foreign"] = Leverage['Foreign_Investor']
stock['Trust'] = Leverage['Investment_Trust']
stock['Dealer'] = Leverage['Dealer_self']+Leverage['Dealer_Hedging']

def condition(row): #[多 空 無]
        import numpy as np
        if row['wave'] > 2:
            if row['down']-row['up'] > 1 or row['gap'] > 1.5:
                return 'u'
            elif row['up']-row['down'] > 1 or row['gap'] < -1.5:
                return 'd'
            else:
                return '-'
        else:
            return '-'

stock['fluctuation'] = stock.apply(condition, axis=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(stock['close'][-50:],label="price", color='red')
plt.grid(True)
plt.twinx()
plt.bar(stock.index[-50:], stock['vol_status'][-50:], label="vol", color=(0.7, 0.5, 0.5), alpha=0.5)
plt.title(f"{stockID}.TW")
plt.legend(loc='upper left')
plt.xticks(stock.index[-50:],stock['fluctuation'][-50:])


plt.subplot(3, 1, 2)
plt.plot(stock['close'][-50:],label="price", color='red')
plt.grid(True)
plt.twinx()
plt.bar(stock.index[-50:], stock['deviations5'][-50:]*10, label="dev5", color=(0, 0, 0.5), alpha=0.5)
plt.title("deviations")
plt.legend()
plt.xticks(stock.index[-50:],stock['fluctuation'][-50:])

# plt.plot(stock['close'][-30:],label="price", color='red')
# plt.xlabel('Date')

# plt.ylabel('price')

plt.show()
