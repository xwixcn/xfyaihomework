#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import random
import json
import time
from PyQt5 import QtWidgets, QtGui
from mainDialog import Ui_Dialog
from fer2013Dataset import FER2013Dataset, pixel_to_image
from train_tensorflow import train, test, MyLoggerCallback
from PyQt5.QtCore import QThread

class MainDialog(QtWidgets.QDialog, Ui_Dialog):
    DEFAULT_CONFIG_FILE = "config-default.json"
    CONFIG_FILE = "config.json"
    EMOTION_SHOW = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

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
        self.updateCnnListView()
        self.updateMlpListView()

        self.saveButton.clicked.connect(self.saveConfig) 
        # load config from file select by user
        self.loadButton.clicked.connect(self.loadConfigDialog)
        self.startButton.clicked.connect(self.doTrain)
        self.testButton.clicked.connect(self.doTest)
        self.continueButton.clicked.connect(self.doContinue)
        self.logcallback = None

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
    
    def updateCnnListView(self):
        self.cnnModel.clear()
        for lcfg in self.config['cnn']['conv_layers']:
            item = QtGui.QStandardItem("Conv2D: " + str(lcfg['filters']) + " " + str(lcfg['kernel_size']))
            self.cnnModel.appendRow(item)

    def updateMlpListView(self):
        self.mlpModel.clear()
        for lcfg in self.config['mlp']['dense_layers']:
            item = QtGui.QStandardItem("Dense: " + str(lcfg['units']) + "  激活函数: " + lcfg['activation'])
            self.mlpModel.appendRow(item)

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
        self.config['mlp']['dense_layers'].append({'units': 128, 'activation': 'relu'})
        self.updateMlpListView()

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
        emotionStr = self.EMOTION_SHOW[emotion]
        X = pixel_to_image(pixels)
        img = QtGui.QImage(X, 48, 48, QtGui.QImage.Format_Indexed8)
        self.imgScene.clear()
        self.imgScene.addPixmap(QtGui.QPixmap.fromImage(img))
        self.graphicsView.fitInView(self.imgScene.itemsBoundingRect())
        self.imgScene.update()
        self.imgIdxLabel.setText(str(idx))
        self.imgEmotionLabel.setText(emotionStr)
        print("show Image done")

    def addLogOutput(self, output):
        self.outputTextBrowser.append(output)
        self.outputTextBrowser.update()
        vscrollBar = self.outputTextBrowser.verticalScrollBar()
        vscrollBar.setValue(vscrollBar.maximum())

    def initThread(self):
        if self.logcallback == None:
            self.logcallback = MyLoggerCallback()
            self.logcallback.setSender(self)
            self.trainThread = TrainThread()
            self.trainThread.setLogCallBack(self.logcallback)
            self.trainThread.setLogOutput(self)
            self.trainThread.start()
    def checkThread(self):
        if self.trainThread.startTraining:
            return "Training is already started"
        if self.trainThread.startTesting:
            return "Testing is already started"
        return None
            
    def doTrain(self):
        self.initThread()
        msg = self.checkThread()
        if msg is not None:
            self.addLogOutput(msg)
            return
        self.trainThread.startTraining = True
        self.trainThread.is_continue = False
        self.outputTextBrowser.clear()
        self.addLogOutput("Start training")
        self.outputTextBrowser.update()
    
    def doContinue(self):
        self.initThread()
        msg = self.checkThread()
        if msg is not None:
            self.addLogOutput(msg)
            return
        self.trainThread.startTraining = True
        self.trainThread.is_continue = True
        self.outputTextBrowser.clear()
        self.addLogOutput("Continue training")
        self.outputTextBrowser.update()

    def doTest(self):
        self.initThread()
        msg = self.checkThread()
        if msg is not None:
            self.addLogOutput(msg)
            return
        self.trainThread.startTesting = True
        self.outputTextBrowser.clear()
        self.addLogOutput("Start testing")
        self.outputTextBrowser.update()

    def setTrainPrecision(self, precision):
        self.config['precision'] = precision

    def updateTrainPrecision(self):
        try:
            showString = "{:.4}".format(self.config['precision']*100)
            self.trainPrecisionLabel.setText(showString)
        except:
            pass
        
    def setTestPrecision(self, precision):
        self.config['precision'] = precision

    def updateTestPrecision(self):
        try:
            showString = "{:.4}".format(self.config['precision']*100)
            self.testPrecisionLabel.setText(showString)
        except:
            pass

class TrainThread(QThread):
    def __init__(self):
        super(QThread, self).__init__()
        self.startTraining = False
        self.startTesting = False
        self.callback = None
        self.output = None
        self.is_continue = False

    def setLogCallBack(self, callback):
        self.callback = callback

    def setLogOutput(self, output):
        self.output = output

    def checkThread(self):
        if self.startTraining:
            return True
        if self.startTesting:
            return True
        return False
    
    def run(self):
        data = None
        while True:
            if not self.checkThread():
                time.sleep(1)
                continue
            if self.startTraining:
                self.output.addLogOutput("Start training")
                if data is None:
                    data = FER2013Dataset(sender=self.output)
                train(self.callback, data, self.is_continue)
                self.startTraining = False
                self.output.addLogOutput("Training done")
            if self.startTesting:
                self.output.addLogOutput("Start testing")
                if data is None:
                    data = FER2013Dataset(sender=self.output)
                test(self.callback, data)
                self.startTesting = False
                self.output.addLogOutput("Testing done")
            time.sleep(1)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = MainDialog()
    Dialog.show()
    sys.exit(app.exec_())
