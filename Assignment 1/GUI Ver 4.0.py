import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np


root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig = Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))


graph = tkinter.Tk()
canvas = FigureCanvasTkAgg(fig, master=graph)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)


canvas.mpl_connect("key_press_event", on_key_press)

def _quit():
    graph.withdraw()

def _open():
    graph.deiconify()     # stops mainloop


button = tkinter.Button(master=graph, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

button1 = tkinter.Button(master=root, text="Open", command=_open)
button1.pack(side=tkinter.TOP)


tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.