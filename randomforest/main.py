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
    
    stockid = input("please enter the stockid: ")
    year = int(input("training year to 2023 from: "))
    print(f"[*] train {stockid}.TW")
    stock = STOCK(stockid, year)
    stock.add_target_info()
    reset = False
    
    
    add_m = input('Do you want to add_moving_average_info? (Y/N)')
    if add_m == 'Y':
        print("[+] add moving average ...")
        stock.add_moving_average_info()
    add_B = input('Do you want to add_BBands_info? (Y/N)')
    if add_B == 'Y':
        print("[+] add BBands info ...")
        stock.add_BBands_info()
    add_L = input('Do you want to add_Leverage? (Y/N)')
    if add_L == 'Y':
        print("[+] add Leverage ...")
        stock.add_Leverage()
    add_M = input('Do you want to add Margin info? (Y/N)')
    if add_L == 'Y':
        print("[+] add Margin info ...")
        stock.add_Margin()
    stock.drop_Nan()
    print(stock.stock)
    print("[*] stock is setup!")
    

    t_or_l = input("construct model? (Y/N)")
    if t_or_l == 'Y':
        model = stock.Forest_model(
            split=100, 
            n_estimators=800, 
            min_samples_split=50,
            depth=8
        )['model']

        print("######### real test: #########")
        test_prodictors = stock.to_test(val=25)[stock.prodictors]
        groundtrue = stock.to_test(val=25)['target'].values
        result_pro = model.predict_proba(test_prodictors)
        result = model.predict(test_prodictors)
        print('predict_pro:', result_pro[:,1].round(2))
        print('[+] predict:    ', result)
        print('[+] ground true:', groundtrue)

        from sklearn.metrics import precision_score, accuracy_score

        precision = precision_score(result, groundtrue, average='weighted')
        accuracy = accuracy_score(result, groundtrue)
        print( f'[+] precision: {precision*100}%',
            f'\n[+] accuracy:  {accuracy*100}%', )
        print()
        do_save = input("do you want to save the model? (Y/N)")
        if do_save == 'Y':
            try:
                joblib.dump(model, f'./randomforest/model/model_{stockid}.joblib')
            except Exception as e:
                print(e)
        else:
            print(f'\n[*] {stockid}--------Done')
    elif t_or_l == 'N':
        print(f'[*] load model from ./randomforest/model/model_{stockid}.joblib')
        try:
            loaded_model = joblib.load(f'./randomforest/model/model_{stockid}.joblib')
            result_pro = loaded_model.predict_proba(stock.predict_tomorrow())
            result = loaded_model.predict(stock.predict_tomorrow())
            pridict_more_preci = result_pro[:,1]
        except Exception as e:
            print(e)
            print(f'\n[*] {stockid}--------Done')
    else:
        print("[*] Error input")

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