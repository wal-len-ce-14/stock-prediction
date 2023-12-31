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
                info = np.array([self.all[i]])
            else:
                info = np.append(info,[self.all[i]], axis=0)

        target = torch.tensor(self.target[index+self.all.shape[1]-1])
        # print(target)
        # change !!!!
        # target = torch.where(target > 0.5, torch.tensor([1,0]), torch.tensor([0,1])) 
        return torch.tensor(info), target
    
class CNN(nn.Module):
    def __init__(self, input=1, output=3, feature=16):
        super(CNN, self).__init__()
        # input output channel
        self.input = input  
        self.output = output
        self.size = feature
        


        # net segment
        self.maxpool = nn.MaxPool2d(2)
        self.fc_pool = nn.AvgPool2d(int(self.size/4), stride=1)
        self.x1init = nn.Conv2d(self.input, 4, 1,1)
        self.x1conv4to4 = nn.Conv2d(4, 4, 1,1)
        self.x3init = nn.Conv2d(4,8, 3,1,1)
        self.x3conv8to8 = nn.Conv2d(8,8, 3,1,1)
        self.x3conv16to16 = nn.Conv2d(16,16,3,1,1, bias=False)
        self.x3conv16to64 = nn.Conv2d(16,64,3,1,1, bias=False)
        self.x3conv64to64 = nn.Conv2d(64,64,3,1,1, bias=False)
        self.x3conv64to128 = nn.Conv2d(64,128,3,1,1, bias=False)
        self.x3conv128to128 = nn.Conv2d(128,128,3,1,1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.init_Neurons = nn.Linear(self.size*self.size*8, self.size*self.size)
        self.f_Neurons = nn.Linear(self.size*self.size, self.size*self.size)
        self.linearNto128 = nn.Linear(self.size*self.size, 128*4)
        self.linear128to128 = nn.Linear(128*4, 128*4)
        self.linear128toout = nn.Linear(128*4, self.output)
        self.normal16 = nn.BatchNorm2d(16)
        self.normal64 = nn.BatchNorm2d(64)
        self.normal128 = nn.BatchNorm2d(128)
        self.d1normal = nn.BatchNorm1d(self.size*self.size)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        output = self.x1init(x)         # ch = 4
        output = self.relu(self.x1conv4to4(output))
        output = self.relu(self.x1conv4to4(output))
        output = self.x3init(output)    # ch = 8
        output = self.relu(self.x3conv8to8(output))
        output = self.relu(self.x3conv8to8(output))

        output = nn.Flatten()(output)
        output = self.init_Neurons(output)
        # 2,4 2,5
        for i in range(0,2):  
            output = self.drop(self.relu(self.f_Neurons(output)))
        output = self.linearNto128(output)
        for i in range(0,4):
            output = self.drop(self.relu(self.linear128to128(output)))

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
        batch=128,
        epochs=100,
        # stockname=0
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    best = 0.45
    #create model
    if if_load != '':
        try:
            model = torch.load(if_load)
            # print(f'Load model from {if_load}')
        except Exception as e:
            print(e)
    else:
        model = CNN(feature=len(prodictor.columns)).to(device=device)
    print("[*] start train CNN")
    print(f"\t with linear model.")
    print(f'\t Load model from {if_load}.')
    print(f"\t feature {len(prodictor.columns)}X{len(prodictor.columns)}")
    print(f"\t batch size {batch}")
    print(f"\t epochs time {epochs}")
    print(f"\t device {device}")
    
    # dataset
    data = Data(prodictor.values, target.values) # .values
    print(f"\t DATA LEN {len(data)}")
    print(f"\t DATA LEN/batch {len(data)/batch}")
    traindata, testdata = random_split(data, [int(len(data)*0.9), len(data)-int(len(data)*0.9)])
    trainLoader = DataLoader(traindata, batch, shuffle=True, drop_last=True)
    testLoader = DataLoader(testdata, batch, shuffle=True, drop_last=True)
    # opt, loss
    # change !!!!
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # loss_f = nn.MSELoss()
    # loss_f = nn.BCELoss()
    loss_f = nn.CrossEntropyLoss()
    loss_plot = []
    test_loss = []
    acc_plot = []
    # train 
    for epoch in range(0, epochs):
        
        epoch_loss = 0
        print(f"[*] epoch {epoch+1}")
        ## training
        from tqdm import tqdm
        for idx, (path, now) in tqdm(enumerate(trainLoader), total=len(trainLoader)):  # path 過去資料 now 現在要預測的
            path = path.to(float32).unsqueeze(1).to(device=device)
            now = now.to(float32).to(device=device)
            pred = torch.sigmoid(model(path))
            loss = loss_f(pred, now)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print(f"\t[+] Batch {idx+1} done, with loss = {loss}")
        print(f"\t[+] train loss = {epoch_loss/len(trainLoader)}")
        loss_plot += [(epoch_loss/len(trainLoader)).detach().to('cpu')]
        # check acc
        # print("\t[*] check accurracy")
        with torch.no_grad():
            all_loss = 0
            acc_all = 0
            for idx, (path, now) in enumerate(testLoader):
                path = path.to(float32).unsqueeze(1).to(device=device)
                now = now.to(float32).to(device=device)
                pred = torch.sigmoid(model(path))

                from sklearn.metrics import accuracy_score
                loss = loss_f(pred, now)
                all_loss += loss
                # print(f'origin \n{(pred[:10], 2)}')
                pred = torch.where(pred > 0.51, 1., 0.)
                acc = accuracy_score(pred.to('cpu'), now.to('cpu'))
                acc_all += acc
                print(f'pred \n{pred[:10].to(int)}')
                print(f'now \n{now[:10].to(int)}')
            print(f'\t[+] test_loss = {(all_loss/len(testLoader))}')  
            print(f'\t[+] acc_all = {(acc_all/len(testLoader))*100}%')
            test_loss += [(all_loss/len(testLoader)).detach().to('cpu')]
            acc_plot += [(acc_all/len(testLoader))]
            if acc_all/len(testLoader) > best:
                print('[*] save this model!!')
                best = acc_all/len(testLoader)
                torch.save(model, f'./model/e{epoch}_{round(best*100, 2)}%.pth')
            print()

    import matplotlib.pyplot as plt
    
    plt.plot(loss_plot, label="Train loss", color='red')
    plt.plot(test_loss,label="Test loss", color='blue')
    plt.grid('True', color='y')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(acc_plot, label="acc", color='red')
    plt.grid('True', color='y')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

    print("[*] train end")
    return model

class Data2(Dataset):
    def __init__(self, all_info, target): 
        self.all = all_info
        self.target = target
    def __len__(self):
        return self.all.shape[0]-20
    def __getitem__(self, index):
        for i in range(index, index+20):
            if i == index:
                info = np.array([self.all[i]])
            else:
                info = np.append(info,[self.all[i]], axis=0)

        target = torch.tensor(self.target[index+20-1])
        # print(target)
        # change !!!!
        # target = torch.where(target > 0.5, torch.tensor([1,0]), torch.tensor([0,1])) 
        return torch.tensor(info), target

class Linear(nn.Module):
    def __init__(self, input=1, output=3, feature=16, day=20):
        super(Linear, self).__init__()
        self.input = input  
        self.output = output
        self.size = feature
        self.day = day
        # net segment
        self.mid = int(self.size*self.day*16)
        self.init = nn.Linear(self.size*self.day, self.mid)
        self.middle = nn.Linear(self.mid, self.mid)
        self.out = nn.Linear(self.mid, self.output)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        output = nn.Flatten()(x)
        output = self.init(output)
        for i in range(0,4):
            output = self.drop(self.relu(self.middle(output)))
        output = self.out(output)
        return output

def Linear_model(
        prodictor,
        target,
        if_load='',
        batch=34,
        epochs=100,
        # stockname=0
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    best = 0.45
    #create model
    if if_load != '':
        try:
            model = torch.load(if_load)
            # print(f'Load model from {if_load}')
        except Exception as e:
            print(e)
    else:
        model = Linear(feature=len(prodictor.columns)).to(device=device)
    print("[*] start train Linear")
    print(f'\t Load model from {if_load}.')
    print(f"\t feature {len(prodictor.columns)} X 30(day)")
    print(f"\t batch size {batch}")
    print(f"\t epochs time {epochs}")
    print(f"\t device {device}")
    
    # dataset
    data = Data2(prodictor.values, target.values) # .values
    print(f"\t DATA LEN {len(data)}")
    print(f"\t DATA LEN/batch {len(data)/batch}")
    traindata, testdata = random_split(data, [int(len(data)*0.9), len(data)-int(len(data)*0.9)])
    trainLoader = DataLoader(traindata, batch, shuffle=True, drop_last=True)
    testLoader = DataLoader(testdata, batch, shuffle=True, drop_last=True)
    # opt, loss
    # change !!!!
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # loss_f = nn.MSELoss()
    # loss_f = nn.BCELoss()
    loss_f = nn.CrossEntropyLoss()
    loss_plot = []
    test_loss = []
    acc_plot = []
    # train 
    for epoch in range(0, epochs):
        
        epoch_loss = 0
        print(f"[*] epoch {epoch+1}")
        ## training
        from tqdm import tqdm
        for idx, (path, now) in tqdm(enumerate(trainLoader), total=len(trainLoader)):  # path 過去資料 now 現在要預測的
            path = path.to(float32).to(device=device)
            now = now.to(float32).to(device=device)
            # print(path.shape, now.shape)
            pred = torch.sigmoid(model(path))
            loss = loss_f(pred, now)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print(f"\t[+] Batch {idx+1} done, with loss = {loss}")
        print(f"\t[+] train loss = {epoch_loss/len(trainLoader)}")
        loss_plot += [(epoch_loss/len(trainLoader)).detach().to('cpu')]
        # check acc
        # print("\t[*] check accurracy")
        with torch.no_grad():
            all_loss = 0
            acc_all = 0
            for idx, (path, now) in enumerate(testLoader):
                path = path.to(float32).to(device=device)
                now = now.to(float32).to(device=device)
                pred = torch.sigmoid(model(path))

                from sklearn.metrics import accuracy_score
                loss = loss_f(pred, now)
                all_loss += loss
                # print(f'origin \n{(pred[:10], 2)}')
                pred = torch.where(pred > 0.51, 1., 0.)
                acc = accuracy_score(pred.to('cpu'), now.to('cpu'))
                acc_all += acc
                if idx == 0:
                    print(f'pred \n{pred[:10].to(int)}')
                    print(f'now \n{now[:10].to(int)}')
            print(f'\t[+] test_loss = {(all_loss/len(testLoader))}')  
            print(f'\t[+] acc_all = {(acc_all/len(testLoader))*100}%')
            test_loss += [(all_loss/len(testLoader)).detach().to('cpu')]
            acc_plot += [(acc_all/len(testLoader))]
            if acc_all/len(testLoader) > best:
                print('[*] save this model!!')
                best = acc_all/len(testLoader)
                torch.save(model, f'./model/e{epoch}_{round(best*100, 2)}%.pth')
            print()
            
    import matplotlib.pyplot as plt
    
    plt.plot(loss_plot, label="Train loss", color='red')
    plt.plot(test_loss,label="Test loss", color='blue')
    plt.grid('True', color='y')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(acc_plot, label="acc", color='red')
    plt.grid('True', color='y')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

    print("[*] train end")
    return model

def usemodel(model_path, stock, bias=0):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path).to(device)
    prodictor, date, ups_or_downs = stock.get(bias)
    # print(prodictor)
    result = torch.sigmoid(model(torch.tensor(prodictor.values, dtype=float32, device=device).unsqueeze(0).unsqueeze(0)))
    # result = torch.sigmoid(model(torch.tensor(prodictor.values, dtype=float32, device=device).unsqueeze(0)))
    result = torch.where(result > 0.501, 1, 0)
    perform = (result.squeeze(0).to('cpu').numpy() == ups_or_downs).sum()
    print(date, 'pred', result.squeeze(0).to('cpu').numpy(), 'true', ups_or_downs, 'perform', perform)
    return result.to('cpu').numpy(), date, ups_or_downs, perform

# x = torch.randn(10,16,20)
# print('in', x.shape)

# model = Linear()
# print('m',model(x).shape)

