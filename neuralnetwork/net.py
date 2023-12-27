import torch
import torch.nn as nn

class Linear_regression(nn.Module):
    def __init__(self, path_data, predict_price):
        super(Linear_regression, self).__init__()
        self.path = path_data
        self.pred = predict_price
        self.fc = nn.Linear(self.path, self.pred)

        self.flat = nn.Flatten(1,2)
        self.fc_manylayer_input = nn.Linear(self.path, 64*2)
        self.fc_manylayer_hidden = nn.Linear(64*2, 64*2)
        self.fc_manylayer_output = nn.Linear(64*2, self.pred)

        self.relu = nn.ReLU()

    def forward(self, x):
        # output = self.fc(x)

        output = self.flat(x)
        output = self.fc_manylayer_input(output)
        output = self.fc_manylayer_hidden(output)
        #####
        output = self.fc_manylayer_output(output)
        
        return output



# x = torch.randn(30, 3).unsqueeze(0)
# model = Linear_regression(30*3, 1)
# print(model(x))