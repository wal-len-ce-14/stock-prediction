import requests as req
from bs4 import BeautifulSoup

stockid = 2330
# url = f'https://goodinfo.tw/tw/ShowBuySaleChart.asp?STOCK_ID={stockid}&CHT_CAT=DATE'
url = 'https://goodinfo.tw/tw/ShowBuySaleChart.asp?STOCK_ID=4173&CHT_CAT=DATE&SHEET=%E4%B8%89%E5%A4%A7%E6%B3%95%E4%BA%BA%E8%B2%B7%E8%B3%A3%E5%BC%B5%E6%95%B8&STEP=DATA&PERIOD=365'
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'}
params = {
    'STOCK_ID': 4173,
    'CHT_CAT': 'DATE',
    'SHEET': '三大法人買賣張數',
    'STEP': 'DATA',
    'PERIOD': 365
}
res = req.post(url, headers=header, params=params)
res.encoding ='utf-8'
soup = BeautifulSoup(res.content, 'lxml')
print(soup)



import pandas as pd

Legal_person_trading = soup.select_one('#divBuySaleDetailData').p.prettify()
print(Legal_person_trading)
# Legal_person_trading_dfs = pd.read_html(Legal_person_trading)
# Legal_person_trading_dfs = Legal_person_trading_dfs[2].set_index('期別')
# print((Legal_person_trading_dfs.tail()))