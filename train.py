import get_stock_info as get
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import no_grad
import numpy as np
import torch.nn as nn
import torch.optim as optim

epochs = 10

def train(
        model,
        stock_id,
        batch_size=10
):
    ## dataset set
    stock = get.Stocks(stock_id)
   
    
    allData = get.Data(stock.get_price_year(), 30)
    Dlen = len(allData)
    train_data, test_data = random_split(allData, [int(Dlen*0.9), Dlen-int(Dlen*0.9)])
    train_Loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_Loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)
    
    ## optimizer, loss_function, ...
    optimizer = optim.Adam(model.parameters())
    loss_f = nn.BCEWithLogitsLoss()
    # loss_f = nn.CrossEntropyLoss()

    ## training epoch
    for epoch in range(1, epochs):
        epoch_loss = 0
        print(f"[*] epoch {epoch}")
        ## training
        for idx, (path, now) in enumerate(train_Loader):
            print(path.shape)
            pred = model(path)
            loss = loss_f(pred, now)
            optimizer.zero_grad()
            loss.backword()
            optimizer.step()
            print(f"\t[+] Batch {idx+1} done, with loss = {loss}")

        ## check accurracy
        print("[*] check accurracy")
        with no_grad():
            for idx, (path, now) in enumerate(test_Loader):
                preds = model(path)
                print(f"now price = \n{now}")
                print(f"pred price = \n{preds}")

