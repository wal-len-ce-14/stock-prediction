import torch
import torch.nn as nn

class Linear_regression(nn.Module):
    def __init__(self, path_data, predict_price):
        super(Linear_regression, self).__init__()
        self.path = path_data
        self.pred = predict_price
        self.fc1 = nn.Linear(self.path, self.pred)

    def forward(self, x):
        print(x.shape)
        return self.fc1(x)



x = torch.randn(10,30)
model = Linear_regression(30, 1)
print(model(x).shape)