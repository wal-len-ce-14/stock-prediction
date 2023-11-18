from train import train 
from net import Linear_regression as ls
# {
#   2344華邦電, 2303聯電, 2388威盛, 2402毅嘉, 3035智原, 2618長榮航, 
# }


period = 30 # day
stockid = 2344
model = ls(period*3, 1)     # 乘3 因為有三個特徵

target_stock = train(model, stockid, period)
test_para, testing_nowprice = target_stock.get_testing_date(period+1)
pred= model(test_para)
print('testing_nowprice = ', testing_nowprice)
print('test_prediction = ', pred)

