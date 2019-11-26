# ___________________________________________
# Data Mining Group 5 Final Project
# Jyoti Sharma, Tanvi Hindwan, and Jessa Henderson
# Features impacting price in Florence Airbnbs
# ___________________________________________

# Import Proper Packages
#WE NEED TO REMOVE ONES THAT AREN'T RELEVANT FOR US AT THE END**

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime, timedelta
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --------------------------------
# Deafault font size for all the windows
# CAN CHANGE IF WE WANT**
# --------------------------------
font_size_window = 'font-size:16px'

#FYI - Each drop down item has a class (here), a button (further below), and a function (below the buttons)
#This follows the demo and tutorials.
#FYI - Below the functions you will find global variables, these need to be adjusted as we move forward to be efficient
#with our code while still pulling in the correct information.
#DELETE THIS NOTE BEFORE SUBMISSION**

#Setup Classes for Each Drop Down Item
class PriceDistribution(QMainWindow):
    send_fig = pyqtSignal(str)

    #--------------------------------------------------------
    # This class if for the price distribution as shown via box plot
    # This is set up using the vertical layout option
    # DO WE WANT TO MANIPULATE THIS DURING THE PRESENTATION? (DROP OUTLIERS?) IF NOT, WE NEED TO ADD UNDER THE FUNCTION.
    # IF WE DO - WE NEED TO ADD MORE HERE.
    #--------------------------------------------------------

    def __init__(self):
        #--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #--------------------------------------------------------
        super(PriceDistribution, self).__init__()

        self.Title = 'Distribution of Florence Airbnb Prices'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout)
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)
        self.label1 = QLabel('This boxplot shows the distribution of Airbnbs in Florence, Italy')
        self.layout.addWidget(self.label1)

        #self.fig = Figure()
        #self.ax1 = self.fig.add_subplot()
        #self.axes = [self.ax1]
        #self.canvas = FigureCanvas(self.fig)

        self.setCentralWidget(self.main_widget)


class MissingData(QMainWindow):
    #::----------------------------------
    # Creates a canvas for the graph of missing data prior to preprocessing
    # COMPLETE AS IS - PUT GRAPH CODE UNDER THE FUNCTION BELOW
    #;;----------------------------------
    def __init__(self, parent=None):
        super(MissingData, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Florence Airbnb Missing Data'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class CorrelationPlot(QMainWindow):
    #::----------------------------------
    # Creates a canvas for the Correlation Plot
    # COMPLETE AS IS - PUT GRAPH CODE UNDER THE FUNCTION BELOW
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CorrelationPlot, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Florence Airbnb Variable Correlation Plot'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class AmenityCount(QMainWindow):
    #::----------------------------------
    # Creates a canvas for the graph on the count by Airbnb amenity
    #COMPLETE AS IS - PUT GRAPH CODE UNDER THE FUNCTION BELOW
    #;;----------------------------------
    def __init__(self, parent=None):
        super(AmenityCount, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Florence Airbnb Count by Amenity'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw plots/graphs
    # this is used by multiple classes
    #COMPLETE AS IS
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

#The next class is for Feature Selection

class RandomForest(QMainWindow):
    send_fig = pyqtSignal(str)

    #--------------------------------------------------------
    # This class if for feature selection with Random Forest
    # This is set up using the vertical layout option (MAY WANT TO CHANGE TO GRID)
    #--------------------------------------------------------

    def __init__(self):
        #--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #--------------------------------------------------------
        super(RandomForest, self).__init__()

        self.Title = 'Feature Selection Using Random Forest'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout)
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)
        self.label1 = QLabel('Random Forest Information Displayed Here')
        self.layout.addWidget(self.label1)

        self.setCentralWidget(self.main_widget)

    #WILL NEED TO BUILD THIS OUT A LOT MORE FOR RANDOM FOREST - LOOK AT DEMO FOR IDEAS RE:CHANGING PARAMETERS IN FRONT OF CLASS

#The next class is for model analysis
class LinearRegression(QMainWindow):
    send_fig = pyqtSignal(str)

    # --------------------------------------------------------
    # This class if for Model Analysis with Linear Regression
    # This is set up using the vertical layout option (MAY WANT TO CHANGE TO GRID)
    # --------------------------------------------------------

    def __init__(self):
        # --------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        # --------------------------------------------------------
        super(LinearRegression, self).__init__()

        self.Title = 'Final Model Analysis'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout)
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)
        self.label1 = QLabel('Linear Regression Information Displayed Here')
        self.layout.addWidget(self.label1)

        self.setCentralWidget(self.main_widget)

    # WILL NEED TO BUILD THIS OUT A LOT MORE FOR LR - LOOK AT DEMO FOR IDEAS RE:CHANGING PARAMETERS IN FRONT OF CLASS

# Setup Main Application
#THIS IS COMPLETE (Unless we want to restyle!)**
class MainWIN(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300
        self.Title = 'Airbnb Prices in Florence, Italy'
        self.setStyleSheet("QWidget {background-image: url(airbnb_logo.png); background-repeat: no-repeat}")
        self.initUI()

    def initUI(self):
        # ::-------------------------------------------------
        # Creates the menu and the items
        # ::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()

        # ::-----------------------------
        # Main Menu Creation
        # ::-----------------------------

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        mainMenu.setStyleSheet('background-color: #FF585D')
        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('Exploratory Analysis')
        FeatureMenu = mainMenu.addMenu('Feature Selection')
        ModelMenu = mainMenu.addMenu('Model Analysis')

        # ::--------------------------------------
        # Exit action
        # ::--------------------------------------

        exitButton = QAction('&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        self.show()

        # ----------------------------------------
        # EDA Buttons
        # Creates the EDA Analysis Menu
        # Price Distribution: Shows the distribution of rental price using a boxplot
        # Missing Data: Shows the amount of data that was missing in the initial dataset
        # Correlation Plot: Correlation plot of Florence Airbnb variables
        # Correlation Bar Graph: Shows the correlation between each individual feature and price
        #ALL BUTTONS ARE COMPLETE!
        #::----------------------------------------

        EDA1Button = QAction('Price Distribution', self)
        EDA1Button.setStatusTip('Boxplot for Price')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction('Missing Data', self)
        EDA2Button.setStatusTip('Missing Data in Initial Dataset')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA3Button = QAction('Correlation Plot', self)
        EDA3Button.setStatusTip('Features Correlation Plot')
        EDA3Button.triggered.connect(self.EDA3)
        EDAMenu.addAction(EDA3Button)

        EDA4Button = QAction('Amenity Count', self)
        EDA4Button.setStatusTip('Bar Graph of Amenity Count')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)

        # ----------------------------------------
        # Feature Selection Button
        # Creates the Feature Selection Drop Down Menu
        # Random Forest: Random Forest was used to determine the best features for the final model.
        # BUTTON IS COMPLETE
        #::----------------------------------------

        FSButton = QAction('Random Forest', self)
        FSButton.setStatusTip('Random Forest for Feature Selection')
        FSButton.triggered.connect(self.FS)
        FeatureMenu.addAction(FSButton)

        # ----------------------------------------
        # Linear Regression Button
        # Creates the Model Analysis Drop Down Menu
        # Linear Regression: Linear Regression was used to investigate how Airbnb features contribute to price
        # BUTTON IS COMPLETE
        #::----------------------------------------

        LRButton = QAction('Linear Regression', self)
        LRButton.setStatusTip('Model Analysis with Linear Regression')
        LRButton.triggered.connect(self.LR)
        ModelMenu.addAction(LRButton)

        #:: Creates an empty list of dialogs to keep track of
        #:: all the iterations
        self.dialogs = list()
        self.show()

    # ----------------------------------------
    # EDA Functions
    # Creates the actions for the EDA Analysis Menu
    # EDA1: Amenity Counts
    # EDA2: Price Distribution
    # EDA3: Correlation Plot
    # EDA4: Correlation Bar Grap
    #::----------------------------------------

    def EDA1(self):
        #::---------------------------------------------------------
        # This function creates an instance of PriceDistribution class
        # This class creates a boxplot that shows price distribution of Florence Airbnbs
        #DO WE WANT TO MANIPULATE THIS IN CLASS? (DROP OUTLIERS?) IF NOT, WE ADD MODEL HERE like EDA2.
        #IF WE DO - WE NEED TO ADD UNDER THE CLASS.
        #::---------------------------------------------------------
        dialog = PriceDistribution()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::----------------------------------------------------------
        # This function creates an instance of the MissingData class
        #NEED TO FIX - NOT CORRECT PLOT (FIX GLOBAL VARIABLES BELOW? ADD FEATURES TO GLOBAL VARIABLES BELOW?)
        #::----------------------------------------------------------

        dialog = MissingData()
        dialog.m.plot()
        dialog.m.ax.bar(plot1, height=100)
        dialog.m.ax.set_title('PLOT1 - Features Emptiness', fontsize=18)
        dialog.m.ax.set_xlabel('Features of Dataset', fontsize=20)
        dialog.m.ax.set_ylabel('Percent Empty Data / NaN', fontsize=15)
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA3(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #NO GRAPH DATA ADDED YET - FOLLOW EDA2 AS EXAMPLE SINCE NOT MANIPULATING PARAMETERS IN FRONT OF CLASS
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the AmenityCount class
        #NO GRAPH DATA ADDED YET - FOLLOW EDA2 AS EXAMPLE SINCE NOT MANIPULATING PARAMETERS IN FRONT OF CLASS
        #::----------------------------------------------------------
        dialog = AmenityCount()
        self.dialogs.append(dialog)
        dialog.show()

    # ----------------------------------------
    # Feature Selection Function
    # Creates the actions for the Feature Selection Menu
    # FS: Feature Selection was conducted using the random forest technique
    #::----------------------------------------

    def FS(self):
        #::----------------------------------------------------------
        # This function creates an instance of the RandomForest class
        #LEAVE AS IS HERE - DEVELOP RANDOM FOREST MODEL INFO UNDER THE CLASS SECTION - LIKE IN THE DEMO
        #::----------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    # ----------------------------------------
    # Linear Regression Function
    # Creates the actions for the Model Analysis Menu
    # LR: Linear Regression was used as the final model for analysis
    #::----------------------------------------

    def LR(self):
        #::----------------------------------------------------------
        # This function creates an instance of the LinearRegression class
        #LEAVE AS IS HERE - DEVELOP RANDOM FOREST MODEL INFO UNDER THE CLASS SECTION - LIKE IN THE DEMO
        #::----------------------------------------------------------
        dialog = LinearRegression()
        self.dialogs.append(dialog)
        dialog.show()

#------------------------
# Application starts here
#------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    mn = MainWIN()
    sys.exit(app.exec_())

#------------------------
# Global variables are below
#------------------------

def data_airbnb():
    #--------------------------------------------------
    # Pulls in data and relevant variables and features.
    # This is needed for the rest of the GUI to produce the proper output.
    # Loads listings.csv (original dataset)
    # Loads airbnb_cleaned.csv (preprocessed dataset)
    # COMMENTED OUT ARE FROM THE DEMO (unless in all caps) - WE MAY OR MAY NOT NEED ITEMS LIKE THAT BASED ON OUR CODE
    #--------------------------------------------------
    global Florencebnb
    global FlorenceFINAL
    global df_plot1
    global plot1
    global df_plot2
    global df_plot3
    #global y
    #global features_list
    #global class_names
    Florencebnb = pd.read_csv('listings.csv')
    #USE JUST ONE OF THE 3? NEED TO FIX THIS TO GET IT TO RUN PROPERLY IN THE GUI
    df_plot1 = Florencebnb.iloc[:, 0:12]
    plot1 = df_plot1.isnull().sum() / Florencebnb.shape[0] * 100
    df_plot2 = Florencebnb.iloc[:, 12:24]
    df_plot3 = Florencebnb.iloc[:, 24:37]
    #y= Florencebnb["Country"]
    FlorenceFINAL = pd.read_csv('airbnb_cleaned.csv')
    #update feature list & class names
    #features_list = ["GDP", "GINI", "VoiceandAccountability", "PoliticalStabilityNoViolence",
         #"GovermentEffectiveness", "RegulatoryQuality", "RuleofLaw", "ControlofCorruption"]
    #class_names = ['Happy', 'Med.Happy', 'Low.Happy', 'Not.Happy']

if __name__ == '__main__':
    data_airbnb()
    main()