import torch
import torch.nn as nn

class Linear_regression(nn.Module):
    def __init__(self, path_data, predict_price):
        super(Linear_regression, self).__init__()
        self.path = path_data
        self.pred = predict_price
        self.fc = nn.Linear(self.path, self.pred)

        self.fc_manylayer_input = nn.Linear(self.path, 60)
        self.fc_manylayer_hidden = nn.Linear(60, 60)
        self.fc_manylayer_output = nn.Linear(60, self.pred)

    def forward(self, x):
        # output = self.fc(x)
        output = self.fc_manylayer_input(x)
        output = self.fc_manylayer_hidden(output)
        #####
        output = self.fc_manylayer_output(output)
        return output



# x = torch.randn(10,30)
# model = Linear_regression(30, 1)
# print(model(x).shape)