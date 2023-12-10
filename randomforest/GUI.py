import tkinter as tk
import threading as thread
import numpy as np

window = tk.Tk()
window.title("決策支援")
window.geometry('480x300')

def deter():
    from net import usemodel
    from setup import STOCK
    stockid = int(stcok_E.get())
    print(f"[*] get {stockid}.TW")
    stock = STOCK(stockid, 2023)
    stock.add_target_info()
    stock.add_moving_average_info()
    stock.add_BBands_info()
    stock.add_Leverage()
    stock.add_Margin()
    stock.drop_Nan()
    print("[*] stock is setup!")
    result, _, _ = usemodel('./randomforest/model/model1.pth', stock, 1) # 2
    print(result)
    if np.array_equal(result,np.array([[1,0]])):
        ans = tk.Label(text="可以", font=('Arial',15,'bold'))
        ans.grid(column=1, row=4, pady=12, ipadx=50, columnspan=1)
    else:
        ans = tk.Label(text="no", font=('Arial',15,'bold'))
        ans.grid(column=1, row=4, pady=12, ipadx=50, columnspan=1)


stock_label = tk.Label(text="股票代號", font=('Arial',15,'bold'))
stock_label.grid(column=0, row=1, pady=(3,8), padx=(15,10),columnspan=1)

stcok_E = tk.Entry(font=('Arial',15))
stcok_E.grid(column=1, row=1, pady=8, ipadx=50,columnspan=3)

predict = tk.Button(text="可以買嗎??", command=deter)
predict.grid(column=1, row=2, pady=12, ipadx=50, columnspan=1)



window.mainloop()