from setup import STOCK
import numpy as np
import warnings
from pandas.errors import PerformanceWarning
import joblib 
# {
#   
# }


all = {}
#       
warnings.filterwarnings("ignore", category=PerformanceWarning)
# for id in [
#     2330, #2330台積電
#     2344, #2344華邦電
#     2301, #2301光寶科
#     2303, #2303聯電
#     2388, #2388威盛
#     2402, #2402毅嘉
#     3035, #3035智原
#     2618, #2618長榮航
#     2313, #2313華通
#     3037, #3037欣興
#     2882, #2882國泰金
#     1513  #1513中興電
#     2408  #2408南亞科
#     2329  #2329華泰
# ]:
while(1):
    print('[*] --setup--') 
    stockid = input("please enter the stockid: ")
    year = int(input("training year to 2023 from: "))
    print(f"[*] train {stockid}.TW")
    stock = STOCK(stockid, year)
    stock.add_target_info()

    stock.add_moving_average_info()
    stock.add_BBands_info()
    stock.add_Leverage()
    stock.add_Margin()
    stock.drop_Nan()
    print("[*] stock is setup!")
    # stock.stock.to_csv(f'./{stockid}from{year}.csv', index=True, sep=',', encoding='utf-8')
    # target_day = input('what day you want to predict? (1,2,5,10) ')
    # model = stock.Forest_model(
    #     split=100, 
    #     n_estimators=800, 
    #     min_samples_split=50,
    #     depth=6,
    #     target_day=target_day
    # )['model']

    from net import CNN_model
    # print(stock.stock['target'].head(10))
    model =  CNN_model(
        stock.stock[stock.prodictors],        
        stock.stock['target']
    )

    again = input('Do you want to try again? (Y/N)')
    if again != 'Y':
        break
    


print('\n[*] --------Over')


# import json
# print(json.dumps(all))
# file = 'test_acc.json'
# with open(file, 'w') as json_file:
#     json.dump(all, json_file)

# print(json.dumps(all))