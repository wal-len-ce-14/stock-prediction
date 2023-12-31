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


def get_Margin_info(stockid, start, end=2023):
    params = {
        "dataset": 'TaiwanStockMarginPurchaseShortSale',
        "data_id": stockid,
        "start_date": f'{start}-01-01',
        "end_date": f'{end}-12-31',
        "token": 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0xMS0yNSAxNToxOTowNyIsInVzZXJfaWQiOiJ3YWxsZW5jZSIsImlwIjoiNjEuNjQuMjguNTYifQ.QUZrY4N6RhlIO90kIDR4YPWQpkAsRZMtLFgW2YJpn_k'
    }
    url = "https://api.finmindtrade.com/api/v4/data"

    data = req.get(url, params=params)
    data = pd.DataFrame(data.json()['data']).drop(columns=['stock_id','MarginPurchaseLimit','ShortSaleLimit'])
    data = data.set_index(data['date'])
    # print(data[['ShortSaleBuy', 'ShortSaleCashRepayment', 'ShortSaleSell', 'ShortSaleTodayBalance']].tail(10))
    data.loc[data['MarginPurchaseYesterdayBalance'] == 0,'MarginPurchaseYesterdayBalance'] = 1
    data.loc[data['ShortSaleYesterdayBalance'] == 0,'ShortSaleYesterdayBalance'] = 1
    data['Margin'] = (data['MarginPurchaseBuy'] - data['MarginPurchaseCashRepayment'] - data['MarginPurchaseSell'])*100 / data['MarginPurchaseYesterdayBalance']
    data['Sale'] = -(data['ShortSaleBuy'] + data['ShortSaleCashRepayment'] - data['ShortSaleSell'])*100 / data['ShortSaleYesterdayBalance']
    return data[['Margin', 'Sale']]

def get_price_detail(stockid, start, end=2023):
    params = {
        "dataset": 'TaiwanStockPrice',
        "data_id": stockid,
        "start_date": f'{start}-01-01',
        "end_date": f'{end}-12-31',
        "token": 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0xMS0yNSAxNToxOTowNyIsInVzZXJfaWQiOiJ3YWxsZW5jZSIsImlwIjoiNjEuNjQuMjguNTYifQ.QUZrY4N6RhlIO90kIDR4YPWQpkAsRZMtLFgW2YJpn_k'
    }
    url = "https://api.finmindtrade.com/api/v4/data"
    data = req.get(url, params=params)
    data = pd.DataFrame(data.json()['data']).drop(columns=['stock_id','Trading_money', 'Trading_turnover', 'spread'])
    data = data.set_index(data['date']).drop(columns=['date'])
    data.rename(columns={'Trading_Volume': 'capacity'}, inplace = True)
    data['wave'] = (data['max']-data['min'])/data['open'] *100
    data['gap'] = (data['close']-data['open'])/data['open'] *100
    data['up'] = (data['max']-data['close'])/(data['open']) *100
    data['down'] = (data['close']-data['min'])/(data['open']) *100
    return data


# get_juridical_person(2330, 2023)
# print(get_price_detail(2330, 2023))

