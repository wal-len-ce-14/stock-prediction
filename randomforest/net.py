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

        target = torch.tensor(self.target[index+self.all.shape[1]-1])
        # change !!!!
        target = torch.where(target > 0.5, torch.tensor([1,0]), torch.tensor([0,1])) 
        return torch.tensor(info), target

class CNN(nn.Module):
    def __init__(self, input=1, output=2, feature=16):
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
        self.d1normal = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        output = self.x1init(x)
        # print('0',output.shape)
        output = self.x3init(output)    # ch = 16
        # print('1',output)
        for i in range(0,4):
            output = self.x3conv16to16(output)
        # output = self.normal16(output)
        output = self.relu(output)
        output = self.x3conv16to64(output)
        output = self.maxpool(output)
        # print('2',output)
        for i in range(0,6):
            output = self.x3conv64to64(output)
            output = self.normal64(output)
            output = self.relu(output)
        output = self.x3conv64to128(output)
        output = self.maxpool(output)
        # print('3',output)
        for i in range(0,7):
            output = self.x3conv128to128(output)
            output = self.normal128(output)
            output = self.relu(output)
        output = self.fc_pool(output)
        output = output.reshape(output.shape[0], -1)
        output = self.linear128to128(output)
        output = self.drop(self.relu(self.linear128to128(output)))
        output = self.relu(self.linear128to128(output))
        output = self.d1normal(output)
        output = self.relu(self.linear128toout(output))
        return output
    
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import float32
def CNN_model(
        prodictor,
        target,
        if_load='',
        batch=64
):
    #create model
    if if_load != '':
        try:
            model = torch.load(if_load)
            print(f'Load model from {if_load}')
        except Exception as e:
            print(e)
    else:
        model = CNN(feature=len(prodictor.columns))
    print("[*] start train CNN")
    
    # dataset
    data = Data(prodictor.values, target.values) # .values
    traindata, testdata = random_split(data, [int(len(data)*0.9), len(data)-int(len(data)*0.9)])
    trainLoader = DataLoader(traindata, batch, shuffle=True, drop_last=True)
    testLoader = DataLoader(testdata, batch, shuffle=True, drop_last=True)
    # opt, loss
    # change !!!!
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_f = nn.BCELoss()

    # train 
    for epoch in range(0, 20):
        epoch_loss = 0
        print(f"[*] epoch {epoch+1}")
        ## training
        for idx, (path, now) in enumerate(trainLoader):  # path 過去資料 now 現在要預測的
            path = path.to(float32).unsqueeze(1)
            now = now.to(float32)
            pred = torch.sigmoid(model(path))
            # target = torch.where(now > 0.5, torch.tensor([1,0], dtype=float32), torch.tensor([0,1], dtype=float32))
            loss = loss_f(pred, now)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\t[+] Batch {idx+1} done, with loss = {loss}")
        print(f"[+] epoch loss = {epoch_loss/len(trainLoader)}")
        # check acc
        print("[*] check accurracy")
        with torch.no_grad():
            best = 0
            acc_all = 0
            for idx, (path, now) in enumerate(testLoader):
                path = path.to(float32).unsqueeze(1)
                now = now.to(float32)
                # target = torch.where(now > 0.5, torch.tensor([1,0], dtype=float32), torch.tensor([0,1], dtype=float32))
                pred = torch.sigmoid(model(path))
                # print(f'pred = \n{pred.view(-1)}')
                pred = torch.where(pred > 0.501, 1, 0)
                
                # iflat = pred.contiguous().view(-1)
                # tflat = now.contiguous().view(-1)
                # print(f"now price = \n{(iflat)}")
                # print(f"pred price = \n{tflat}")
                # intersection = (iflat * tflat).sum()
                # dice = (2.0 * intersection + 1) / (iflat.sum() + tflat.sum() + 1)
                # acc = (iflat == tflat).sum() / now.numel()
                # print('[+] dice =', dice)

                # print(f'now = \n{now.to(int)}')
                # print(f'pred = \n{pred}')
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(pred, now)
                acc_all += acc
                print(f'[+] acc = {round(acc*100, 2)}%', )
            if acc_all/len(testLoader) > best:
                print('save this model!!')
                best = acc_all/len(testLoader)
                torch.save(model, f'./randomforest/model/model1.pth')
            print()

    print("[*] train end")
    return model
                

# x = torch.randn(10,1,16,16)
# print('in', x.shape)

# model = CNN()
# print('m',model(x).shape)

