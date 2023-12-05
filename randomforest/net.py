import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class Data(Dataset):
    def __init__(self, all_info, target): 
        self.all = all_info
        self.target = target
    def __len__(self):
        return self.all.shape[0]-self.all.shape[1]
    def __getitem__(self, index):
        for i in range(index, index+self.all.shape[1]):
            if i == index:
                info = [self.all[i]]
            else:
                info += [self.all[i]]

        return torch.tensor(info), torch.tensor(self.target[index])

class CNN(nn.Module):
    def __init__(self, input=1, output=1, feature=16):
        super(CNN, self).__init__()
        # input output channel
        self.input = input  
        self.output = output
        self.size = feature

        # net segment
        self.maxpool = nn.MaxPool2d(2)
        self.fc_pool = nn.AvgPool2d(int(self.size/4), stride=1)
        self.x1init = nn.Conv2d(self.input, 8, 1,1)
        self.x3init = nn.Conv2d(8, 16, 3,1,1)
        self.x3conv16to16 = nn.Conv2d(16,16,3,1,1, bias=False)
        self.x3conv16to64 = nn.Conv2d(16,64,3,1,1, bias=False)
        self.x3conv64to64 = nn.Conv2d(64,64,3,1,1, bias=False)
        self.x3conv64to128 = nn.Conv2d(64,128,3,1,1, bias=False)
        self.x3conv128to128 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.linear128to128 = nn.Linear(128, 128)
        self.linear128toout = nn.Linear(128, self.output)
        self.normal16 = nn.BatchNorm2d(16)
        self.normal64 = nn.BatchNorm2d(64)
        self.normal128 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        output = self.x1init(x)
        # print('0',output.shape)
        output = self.x3init(output)    # ch = 16
        # print('1',output)
        for i in range(0,1):
            output = self.x3conv16to16(output)
        # output = self.normal16(output)
        output = self.x3conv16to64(output)
        output = self.maxpool(output)
        # print('2',output)
        for i in range(0,1):
            output = self.x3conv64to64(output)
            # output = self.normal64(output)
        output = self.x3conv64to128(output)
        output = self.maxpool(output)
        # print('3',output)
        for i in range(0,1):
            output = self.x3conv128to128(output)
            # output = self.normal128(output)
        output = self.fc_pool(output)
        output = output.reshape(output.shape[0], -1)
        output = self.linear128to128(output)
        output = self.drop(self.relu(self.linear128to128(output)))
        output = self.relu(self.linear128to128(output))
        output = (self.linear128toout(output))
        return output
    
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import float32
def CNN_model(
        prodictor,
        target,
        batch=16
):
    print("[*] start train CNN")
    #create model
    model = CNN(feature=len(prodictor.columns))
    # dataset
    data = Data(prodictor.values, target.values) # .values
    traindata, testdata = random_split(data, [int(len(data)*0.9), len(data)-int(len(data)*0.9)])
    trainLoader = DataLoader(traindata, batch, shuffle=True, drop_last=True)
    testLoader = DataLoader(testdata, batch, shuffle=True, drop_last=True)
    # opt, loss
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_f = nn.MSELoss()

    # train 
    for epoch in range(0, 20):
        epoch_loss = 0
        print(f"[*] epoch {epoch+1}")
        ## training
        for idx, (path, now) in enumerate(trainLoader):  # path 過去資料 now 現在要預測的
            path = path.to(float32).unsqueeze(1)
            now = now.to(float32).unsqueeze(1)
            pred = model(path)
            loss = loss_f(pred, now)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\t[+] Batch {idx+1} done, with loss = {loss}")
        print(f"[+] epoch loss = {epoch_loss/idx}")
        # check acc
        print("[*] check accurracy")
        with torch.no_grad():
            for idx, (path, now) in enumerate(testLoader):
                path = path.to(float32).unsqueeze(1)
                now = now.to(float32).unsqueeze(0)
                print(path.shape, now.shape)
                pred = torch.sigmoid(model(path))
                pred = torch.where(pred > 0.5, 1, 0)
                
                iflat = pred.contiguous().view(-1)
                tflat = now.contiguous().view(-1)
                print(f"now price = \n{(iflat[-10:]).to(int)}")
                print(f"pred price = \n{tflat[-10:]}")
                intersection = (iflat * tflat).sum()
                dice = (2.0 * intersection + 1) / (iflat.sum() + tflat.sum() + 1)
                acc = (iflat == tflat).sum() / now.numel()
                print('[+] dice =', dice)
                print('[+] acc =', acc)
                print()

    print("[*] train end")
    return model
                

x = torch.randn(10,1,16,16)
print('in', x.shape)

model = CNN()
print('m',model(x).shape)

