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
        # print(target)
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
        self.f_Neurons = nn.Linear(self.size*self.size, self.size*self.size)
        self.linearNto128 = nn.Linear(self.size*self.size, 128*4)
        self.linear128to128 = nn.Linear(128*4, 128*4)
        self.linear128toout = nn.Linear(128*4, self.output)
        self.normal16 = nn.BatchNorm2d(16)
        self.normal64 = nn.BatchNorm2d(64)
        self.normal128 = nn.BatchNorm2d(128)
        self.d1normal = nn.BatchNorm1d(self.size*self.size)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        # output = self.x1init(x)
        # # print('0',output.shape)
        # output = self.x3init(output)    # ch = 16
        # # print('1',output)
        # res = output
        # for i in range(0,6):
        #     output = self.x3conv16to16(output)
        # output = self.normal16(output)
        # output = self.relu(output)
        # # output = torch.cat([output, res], dim=1)
        # # output = nn.Conv2d(32,16,3,1,1)(output)
        # output = self.x3conv16to64(output)
        # output = self.maxpool(output)

        # res = output
        # for i in range(0,6):
        #     output = self.x3conv64to64(output)
        #     output = self.normal64(output)
        #     output = self.relu(output)
        # # output = torch.cat([output, res], dim=1)
        # # output = nn.Conv2d(128,64,3,1,1)(output)
        # output = self.x3conv64to128(output)
        # output = self.maxpool(output)
        # # print('3',output)
        # res = output
        # for i in range(0,7):
        #     output = self.x3conv128to128(output)
        #     output = self.normal128(output)
        #     output = self.relu(output)
        # output = self.fc_pool(output)
        # output = output.reshape(output.shape[0], -1)
        output = nn.Flatten()(x)
        for i in range(0,8):
            output = self.relu(self.f_Neurons(output))
            # if(output.shape[0]):
            #     output = self.d1normal(output)
        output = self.linearNto128(output)
        for i in range(0,6):
            output = self.drop(self.relu(self.linear128to128(output)))
            # output = self.d1normal(output)
        # output = self.d1normal(output)
        output = self.linear128toout(output)
        return output
    
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import float32
def CNN_model(
        prodictor,
        target,
        if_load='',
        batch=40
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    best = 0
    #create model
    if if_load != '':
        try:
            model = torch.load(if_load)
            print(f'Load model from {if_load}')
        except Exception as e:
            print(e)
    else:
        model = CNN(feature=len(prodictor.columns)).to(device=device)
    print("[*] start train CNN")
    
    # dataset
    data = Data(prodictor.values, target.values) # .values
    traindata, testdata = random_split(data, [int(len(data)*0.9), len(data)-int(len(data)*0.9)])
    trainLoader = DataLoader(traindata, batch, shuffle=True, drop_last=True)
    testLoader = DataLoader(testdata, batch, shuffle=True, drop_last=True)
    # opt, loss
    # change !!!!
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # loss_f = nn.MSELoss()
    # loss_f = nn.BCELoss()
    loss_f = nn.CrossEntropyLoss()

    # train 
    for epoch in range(0, 60):
        
        epoch_loss = 0
        print(f"[*] epoch {epoch+1}")
        ## training
        for idx, (path, now) in enumerate(trainLoader):  # path 過去資料 now 現在要預測的
            path = path.to(float32).unsqueeze(1).to(device=device)
            now = now.to(float32).to(device=device)
            # print(now[10:])
            pred = torch.sigmoid(model(path))
            loss = loss_f(pred, now)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"\t[+] Batch {idx+1} done, with loss = {loss}")
        print(f"[+] epoch loss = {epoch_loss/len(trainLoader)}")
        # check acc
        print("\t[*] check accurracy")
        with torch.no_grad():
            all_loss = 0
            acc_all = 0
            for idx, (path, now) in enumerate(testLoader):
                path = path.to(float32).unsqueeze(1).to(device=device)
                now = now.to(float32).to(device=device)
                pred = torch.sigmoid(model(path))
                
                pred = torch.where(pred > 0.501, 1., 0.)
                # print(f'pred = \n{pred[-10:].to(int)}\nnow = \n{now[-10:].to(int)}')

                from sklearn.metrics import accuracy_score
                loss = loss_f(now, pred)
                all_loss += loss
                acc = accuracy_score(pred.to('cpu'), now.to('cpu'))
                acc_all += acc
                # print(f'loss = {loss}')
                
            print(f'[+] acc_all = {(acc_all/len(testLoader))*100}%')
            print(f'[+] all_loss = {(all_loss/len(testLoader))}')
            if round(acc_all/len(testLoader), 0) > best:
                print('save this model!!')
                best = acc_all/len(testLoader)
                torch.save(model, f'./randomforest/model/e{epoch}_{round(best*100, 2)}%.pth')
            print()

    print("[*] train end")
    return model

def usemodel(model_path, stock, bias=0):
    model = torch.load(model_path)
    prodictor, date, ups_or_downs = stock.get(bias)
    # print(prodictor)
    result = torch.sigmoid(model(torch.tensor(prodictor.values, dtype=float32).unsqueeze(0).unsqueeze(0)))
    result = torch.where(result > 0.501, 1, 0)
    print(result, date, ups_or_downs)
    return result.numpy(), date, ups_or_downs

# x = torch.randn(10,1,16,16)
# print('in', x.shape)

# model = CNN()
# print('m',model(x).shape)

