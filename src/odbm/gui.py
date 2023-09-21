from tkinter import *
from os.path import exists
import pandas as pd
from odbm_main import ModelBuilder, ModelHandler
from odbm.utils import extractParams, fmt


# Main Window - load model
class MainWindow:
    def __init__(self,master):
        self.master = master
        self.welcome = Label(master, text = "Welcome to ODBM \n Please insert file path for model to load:")
        self.welcome.config(font = ('Arial', 12))

        self.entry = Entry(master)
        self.loadbutton = Button(master, text="Load Model", fg="blue", command = self.loadModel(self.entry))

        self.welcome.pack()
        self.loadbutton.pack(side = BOTTOM)
        self.entry.pack()


    # Checks if model path given by user exists and builds model.
    def loadModel(model_path,root):
        global myModel #create global variable so it is accessible in main
        if exists(model_path) and model_path.endswith(('.xls','.xlsx')):
            model_species = pd.read_excel(model_path, sheet_name = 'Species & Base Mechanisms', engine = 'openpyxl').dropna('index','all')
            model_rxns = pd.read_excel(model_path, sheet_name = 'Reaction', engine = 'openpyxl').dropna('index','all')
            myModel = ModelBuilder(model_species, model_rxns)
            print('Model successfully built')
            modelWindow(myModel, model_path, root)

        else:
            if exists(model_path):
                raise TypeError('Please pass in an excel compatible workbook with sheets named "Species & Base Mechanisms" and "Reaction"')
            else:
                raise FileExistsError('Error: No file exists with given name.')

# Model Window - back, simulate, or edit
class ModelWindow:
    def __init__(self,master):
        return
# Simulation Window - set simulation parameters, run simulation, save and visualize output
class SimulationWindow:
    def __init__(self,master):
        return
# Edit Window - set boundary species, add event, run sensitivity analysis and visualize output
class EditWindow:
    def __init__(self,master):
        return

root = Tk()
main_window = MainWindow(root)
root.mainloop()