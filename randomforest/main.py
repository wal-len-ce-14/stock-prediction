from setup import STOCK
import numpy as np
import warnings
from pandas.errors import PerformanceWarning
# {
#   
# }


all = {}
#       
warnings.filterwarnings("ignore", category=PerformanceWarning)
for id in [
    # 2330, #2330台積電
    # 2344, #2344華邦電
    # 2301, #2301光寶科
    # 2303, #2303聯電
    # 2388, #2388威盛
    # 2402, #2402毅嘉
    # 3035, #3035智原
    # 2618, #2618長榮航
    # 2313, #2313華通
    # 3037, #3037欣興
    # 2882, #2882國泰金
    1513  #1513中興電
]:
    stockid = id

    print(f"[*] train {stockid}.TW")

    stock = STOCK(stockid, 2020)
    stock.add_target_info()
    stock.add_moving_average_info()
    stock.add_BBands_info()
    stock.add_Leverage()
    stock.drop_Nan()
    # # print("stock:\n",stock.stock['Dealer_Hedging'])
    # # break
    # model = stock.Forest_model(
    #     split=100, 
    #     n_estimators=700, 
    #     min_samples_split=70,
    #     depth=4
    # )['model']

    # print("#########")
    # print('real test:')
    # test_prodictors = stock.to_test(val=25)[stock.prodictors]
    # groundtrue = stock.to_test(val=25)['target'].values
    # result_pro = model.predict_proba(test_prodictors)
    # result = model.predict(test_prodictors)

    # # print('predict_pro:\n', result_pro.round(2))
    # print('predict:    ', result)
    # print('ground true:', groundtrue)

    # from sklearn.metrics import precision_score, accuracy_score

    # precision = precision_score(result, groundtrue)
    # accuracy = accuracy_score(result, groundtrue)
    # print(f'precision', precision,'\naccuracy', accuracy)
    # print()

    import joblib
    # joblib.dump(model, f'./randomforest/model/model_{stockid}.joblib')
    print(stock.to_test(val=10))
    loaded_model = joblib.load(f'./randomforest/model/model_{stockid}.joblib')
    result_pro = loaded_model.predict_proba(stock.predict_tomorrow())
    result = loaded_model.predict(stock.predict_tomorrow())
    pridict_more_preci = result_pro[:,1]


    print('predict_pro:\n', result_pro[:,1].round(2))

    pridict_more_preci[pridict_more_preci > 0.52] = 1
    pridict_more_preci[pridict_more_preci <= 0.52] = 0
    print('more_preci :', pridict_more_preci.astype(int))
    print('predict:    ', result)
    # print('ground true:', groundtrue)

    from sklearn.metrics import precision_score, accuracy_score

    # precision = precision_score(result, groundtrue)
    # accuracy = accuracy_score(result, groundtrue)
    # print(f'precision', precision,'\naccuracy', accuracy)

    # precision = precision_score(pridict_more_preci.astype(int), groundtrue)
    # accuracy = accuracy_score(pridict_more_preci.astype(int), groundtrue)
    # print(f'precision_pro', precision,'\naccuracy_pro', accuracy)


    # best = 0
    # best_t = []
    # best_inreal = 0
    # best_inreal_t =[]
    
    # for n in range(100, 2000, 100):
    #     print(f"###### n_estimators={n} ######")
    #     for min in range(10, 100, 10):
    #         if n < min:
    #             pass
    #         acc = stock.Forest_model(split=100,n_estimators=n, min_samples_split=min)
    #         if acc['accuracy'] > best:
    #             best = acc['accuracy'] 
    #             best_t = [n,min]

    #         test_prodictors = stock.to_test(val=50)[stock.prodictors]
    #         groundtrue = stock.to_test(val=50)['target'].values
    #         result = acc['model'].predict(test_prodictors)

    #         precision = precision_score(result, groundtrue)
    #         accuracy = accuracy_score(result, groundtrue)
    #         print(f'precision', precision,'\naccuracy', accuracy)
    #         if accuracy > best_inreal:
    #             best_inreal = accuracy 
    #             best_inreal_t = [n,min]



    # print(f"########### {stockid }best ###########\n")
    # print('best', best)
    # print('best_t', best_t)
    # print('best_inreal',best_inreal)
    # print('best_inreal_t', best_inreal_t)
    # print(f"########### ############# ############\n")

#     new_record =  {
#         f'stocks_{stockid}': {
#             'prec': precision,
#             'acc': accuracy
            
#             # 'best': best,
#             # 'best_t': best_t,
#             # 'best_inreal': best_inreal,
#             # 'best_inreal_t': best_inreal_t
#         }
#     }
#     all.update(new_record)

# import json
# print(json.dumps(all))
# file = 'test_acc.json'
# with open(file, 'w') as json_file:
#     json.dump(all, json_file)

# print(json.dumps(all))