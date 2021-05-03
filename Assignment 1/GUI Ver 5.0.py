import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np

x = range(0, 10, 2)
y = x

def plot1():
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    plt.show() 
    
def plot2():
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    plt.show()     

def plot3():
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    plt.show()     

def plot4():
    plt.figure("test")
    plt.plot(x,y)    
    plt.show() 

def close_fig():
    plt.close()


root = tk.Tk()
button = tk.Button(root, text='close plot', command = close_fig)
button.grid(row=0, column=0)
button2 = tk.Button(root, text='open plot 1', command = plot1)
button2.grid(row=0, column=1)
button3 = tk.Button(root, text='open plot 2', command = plot2)
button3.grid(row=0, column=2)
button4 = tk.Button(root, text='open plot 3', command = plot3)
button4.grid(row=0, column=3)
button5 = tk.Button(root, text='open plot 4', command = plot4)
button5.grid(row=0, column=4)
root.mainloop()