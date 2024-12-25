#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import random
import json
from PyQt5 import QtWidgets, QtGui
from mainDialog import Ui_Dialog
from fer2013Dataset import FER2013Dataset, pixel_to_image

class MainDialog(QtWidgets.QDialog, Ui_Dialog):
    DEFAULT_CONFIG_FILE = "config-default.json"
    CONFIG_FILE = "config.json"

    def __init__(self):
        super(MainDialog, self).__init__()
        self.setupUi(self)
        self.loadConfig()
        self.mlpListView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cnnListView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.editMlpButton.clicked.connect(self.addMLP) # type: ignore
        self.editCNNButton.clicked.connect(self.addCNN) # type: ignore
        self.mlpListView.doubleClicked['QModelIndex'].connect(self.editMLP) # type: ignore
        self.cnnListView.doubleClicked['QModelIndex'].connect(self.editCNN) # type: ignore

        self.typeRadioButtonMLP = self.createRadioButton("MLP")
        self.typeRadioButtonCNN = self.createRadioButton("CNN")
        self.typeLayout.addWidget(self.typeRadioButtonMLP)
        self.typeLayout.addWidget(self.typeRadioButtonCNN)
        self.updateType()
        
        self.optimizerRadioButtonSGD = self.createRadioButton("SGD")
        self.optimizerRadioButtonAdam = self.createRadioButton("Adam")
        self.optimizerLayout.addWidget(self.optimizerRadioButtonSGD)
        self.optimizerLayout.addWidget(self.optimizerRadioButtonAdam)
        self.updateOptimizer()

        self.updateLearningRate()
        self.updateMaxIter()
        self.updateCheckpoint()

        self.lineEdit.textChanged.connect(lambda text: self.setLearningRate(float(text)))
        self.lineEdit_2.textChanged.connect(lambda text: self.config.__setitem__('max_iter', int(text)))
        self.lineEdit_3.textChanged.connect(lambda text: self.config.__setitem__('checkpoint', int(text)))

        self.mlpModel = QtGui.QStandardItemModel()
        self.mlpListView.setModel(self.mlpModel)
        self.cnnModel = QtGui.QStandardItemModel()
        self.cnnListView.setModel(self.cnnModel)
        self.showImgButton.clicked.connect(self.showImage)
        self.imgScene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.imgScene)
        self.data = FER2013Dataset(max_samples=100)

        self.saveButton.clicked.connect(self.saveConfig) 
        # load config from file select by user
        self.loadButton.clicked.connect(self.loadConfigDialog)

    def loadConfigDialog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Config File", "", "JSON Files (*.json)")
        if filename:
            self.loadConfig(filename)
            self.updateType()
            self.updateOptimizer()
            self.updateLearningRate()
            self.updateMaxIter()
            self.updateCheckpoint()
            
    def loadConfig(self, filename=None):
        config_filename = None
        if filename is not None:
            config_filename = filename
        else:
            config_filename = self.CONFIG_FILE

        try:
            with open(config_filename, "r") as f:
                self.config = json.load(f)
        except:
            with open(self.DEFAULT_CONFIG_FILE, "r") as f:
                self.config = json.load(f)

    def saveConfig(self):
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)
    
    def setType(self, type):
        self.config['type'] = type

    def updateType(self):
        if self.config['type'] == "MLP":
            self.typeRadioButtonMLP.setChecked(True)
        else:
            self.typeRadioButtonCNN.setChecked(True)

    def setOptimizer(self, optimizer):
        self.config['optimizer'] = optimizer

    def setLearningRate(self, learning_rate):
        self.config['learning_rate'] = learning_rate

    def updateLearningRate(self):
        self.lineEdit.setText(str(self.config['learning_rate']))

    def updateOptimizer(self):
        if self.config['optimizer'] == "SGD":
            self.optimizerRadioButtonSGD.setChecked(True)
        else:
            self.optimizerRadioButtonAdam.setChecked(True)

    def updateMaxIter(self):
        self.lineEdit_2.setText(str(self.config['max_iter']))

    def updateCheckpoint(self):
        self.lineEdit_3.setText(str(self.config['checkpoint']))

    def createRadioButton(self, text):
        radioButton = QtWidgets.QRadioButton(text)
        radioButton.toggled.connect(self.onRadioButtonToggled)
        return radioButton
    
    def onRadioButtonToggled(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            print(radioButton.text())
            self.setType(radioButton.text())

    def addMLP(self):
        print("addMLP")
        # add a line to self.listView.
        item = QtGui.QStandardItem("隐含层: 100")
        self.mlpModel.appendRow(item)

    def editMLP(self, index):
        print("editMLP")
        # edit the line of index in self.listView.
        item = self.mlpModel.itemFromIndex(index)
        print(item.text())
        return True

    def addCNN(self):
        print("addCNN")

    def editCNN(self, index):
        print("editCNN")
        
    def showImage(self):
        size = len(self.data)
        idx = random.randint(0, size - 1)
        x = self.data[idx]
        pixels = x[1]
        emotion = x[0]
        X = pixel_to_image(pixels)
        img = QtGui.QImage(X, 48, 48, QtGui.QImage.Format_Indexed8)
        self.imgScene.clear()
        self.imgScene.addPixmap(QtGui.QPixmap.fromImage(img))
        self.graphicsView.fitInView(self.imgScene.itemsBoundingRect())
        self.imgScene.update()
        self.imgIdxLabel.setText(str(idx))
        print("done")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = MainDialog()
    Dialog.show()
    sys.exit(app.exec_())
