import tkinter as tk
import threading as thread

window = tk.Tk()
window.title("決策支援")
window.geometry('480x300')

def deter():
    
    if stcok_E.get() != '':
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