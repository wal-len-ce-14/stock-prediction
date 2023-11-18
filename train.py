import torch
import numpy as np
import get_stock_info as get
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from torch import float32
import torch.optim as optim

epochs = 50

def train(
        model,
        stock_id,
        period,
        batch_size=20
):
    ## dataset set
    stock = get.Stocks(stock_id)
    allData = get.Data(stock.get_price_year(), period)
    # allData = get.Data(stock.get_price_year(), 30)
    Dlen = len(allData)
    train_data, test_data = random_split(allData, [int(Dlen*0.9), Dlen-int(Dlen*0.9)])
    train_Loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_Loader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True)
    

    ## optimizer, loss_function, ...
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss_f = nn.BCEWithLogitsLoss()
    # loss_f = nn.CrossEntropyLoss()
    loss_f = nn.MSELoss()

    ## training epoch
    for epoch in range(0, epochs):
        epoch_loss = 0
        print(f"[*] epoch {epoch+1}")
        ## training
        for idx, (path, now) in enumerate(train_Loader):
            path = path.to(float32)
            now = now.to(float32).unsqueeze(0)
            pred = model(path)
            loss = loss_f(pred, now)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\t[+] Batch {idx+1} done, with loss = {loss}")
        print(f"[+] epoch loss = {epoch_loss/idx}")
        ## check accurracy

        print("[*] check accurracy")
        with torch.no_grad():
            for idx, (path, now) in enumerate(test_Loader):
                path = path.to(float32)
                now = now.to(float32).unsqueeze(1)
                preds = model(path)
                # print(f"now price = \n{now}")
                # print(f"pred price = \n{preds}")
                print(f"\t[+] diff = +-{np.round(torch.sum(abs(now-preds) / now ) / batch_size, 2)}%")
    print(f"recent stock info {stock.get_recent()}")
    return stock

