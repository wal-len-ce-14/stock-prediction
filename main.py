from train import train 
from net import Linear_regression as ls

period = 30 # day
stockid = 2344 # 2344華邦電
model = ls(period, 1)
train(model, stockid)


