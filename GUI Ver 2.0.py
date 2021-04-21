import tkinter as Tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

########################################################################
class MyApp(object):
    """"""
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        self.root = parent
        self.root.title("Main frame")
        self.frame = Tk.Frame(parent)
        self.frame.pack()
                
        btn1 = Tk.Button(self.frame, text="Open Frame", command=self.openFrame)
        btn2 = Tk.Button(self.frame, text="Open Frame", command=self.openFrame)
        btn1.pack()
        btn2.pack()
        
    #----------------------------------------------------------------------
    def hide(self):
        """"""
        self.root.withdraw()
        
    #----------------------------------------------------------------------
    def openFrame(self):
        """"""
        self.hide()
        otherFrame = Tk.Toplevel()
        otherFrame.geometry("1000x800")
        otherFrame.title("otherFrame")
        handler = lambda: self.onCloseOtherFrame(otherFrame)
        btn = Tk.Button(otherFrame, text="Close", command=handler)
        
        pls = self.plot1()
        btn2 = Tk.Button(otherFrame, 
                     command = pls,
                     height = 2, 
                     width = 10,
                     text = "Plot 1")
        btn.pack()
        btn2.pack()
        
    #----------------------------------------------------------------------
    def onCloseOtherFrame(self, otherFrame):
        """"""
        otherFrame.destroy()
        self.show()
        
    #----------------------------------------------------------------------
    def show(self):
        """"""
        self.root.update()
        self.root.deiconify()
    
    def plot1(self):
  
        # the figure that will contain the plot
        fig = Figure(figsize = (6, 6),
                     dpi = 100)
      
        # list of squares // Data
        y = [i**2 for i in range(101)]
      
        # adding the subplot
        plot1 = fig.add_subplot(111)
      
        # plotting the graph
        plot1.plot(y)
      
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master = root)  
        canvas.draw()
      
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()
      
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
      
        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()
        
    
#----------------------------------------------------------------------
if __name__ == "__main__":
    root = Tk.Tk()
    root.geometry("1000x800")
    app = MyApp(root)
    root.mainloop()