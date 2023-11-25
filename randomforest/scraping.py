import requests as req
from bs4 import BeautifulSoup
from urllib import parse
from datetime import date, timedelta
import pandas as pd

# token:
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0xMS0yNSAxNToxOTowNyIsInVzZXJfaWQiOiJ3YWxsZW5jZSIsImlwIjoiNjEuNjQuMjguNTYifQ.QUZrY4N6RhlIO90kIDR4YPWQpkAsRZMtLFgW2YJpn_k

def get_juridical_person(stockid, start, end=2023):
    params = {
        "dataset": 'TaiwanStockInstitutionalInvestorsBuySell',
        "data_id": stockid,
        "start_date": f'{start}-01-01',
        "end_date": f'{end}-12-31',
        "token": 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0xMS0yNSAxNToxOTowNyIsInVzZXJfaWQiOiJ3YWxsZW5jZSIsImlwIjoiNjEuNjQuMjguNTYifQ.QUZrY4N6RhlIO90kIDR4YPWQpkAsRZMtLFgW2YJpn_k'
    }
    url = "https://api.finmindtrade.com/api/v4/data"

    data = req.get(url, params=params)
    data = pd.DataFrame(data.json()['data']).drop(columns=['stock_id'])
    data = data.set_index(data['date'])
    data = data.reindex(columns=['name', 'buy', 'sell'])
    data['total'] = data['buy'] - data['sell']
    Investment_Trust = data[data['name'] == 'Investment_Trust']
    Foreign_Investor = data[data['name'] == 'Foreign_Investor']
    Dealer_Hedging = data[data['name'] == 'Dealer_Hedging']
    Dealer_self = data[data['name'] == 'Dealer_self']
    

    Leverage = pd.concat([Foreign_Investor['total'], 
                          Investment_Trust['total'], 
                          Dealer_self['total'],
                          Dealer_Hedging['total']], 
                          axis=1, 
                          keys=['Foreign_Investor', 
                                'Investment_Trust', 
                                'Dealer_self',
                                'Dealer_Hedging'])

    return Leverage


# print(get_juridical_person(2330, 2023).tail())

