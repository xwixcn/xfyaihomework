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
from trainThread import TrainThread

class MainDialog(QtWidgets.QDialog, Ui_Dialog):
    DEFAULT_CONFIG_FILE = "config-default.json"
    CONFIG_FILE = "config.json"
    EMOTION_SHOW = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self):
        super(MainDialog, self).__init__()
        self.setupUi(self)
        self.loadConfig()

        self.addMlpButton.clicked.connect(self.addMLP) 
        self.addCNNButton.clicked.connect(self.addCNN) 
        self.removeMlpButton.clicked.connect(self.removeMLP)
        self.removeCNNButton.clicked.connect(self.removeCNN)
        self.editMlpButton.clicked.connect(self.confirmEditMLP)
        self.editCNNButton.clicked.connect(self.confirmEditCNN)
        self.mlpListView.doubleClicked['QModelIndex'].connect(self.editMLP) 
        self.cnnListView.doubleClicked['QModelIndex'].connect(self.editCNN)

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

        self.frameworkRadioButtonTensorflow = self.createRadioButton("Tensorflow")
        self.frameworkRadioButtonKeras = self.createRadioButton("Keras")
        self.frameworkLayout.addWidget(self.frameworkRadioButtonTensorflow)
        self.frameworkLayout.addWidget(self.frameworkRadioButtonKeras)
        self.updateFramework()

        self.updateLearningRate()
        self.updateMaxIter()
        self.updateCheckpoint()

        self.learnRateEdit.textChanged.connect(self.setLearningRate)
        self.maxEpocsEdit.textChanged.connect(self.setMaxIter)
        self.autoSaveEpocsEdit.textChanged.connect(lambda text: self.config.__setitem__('checkpoint', int(text)))

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

        self.initThread()

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
            typeStr = lcfg['type']
            item = QtGui.QStandardItem(typeStr + ": " + str(lcfg['filters']) + " " + str(lcfg['kernel_size']) + ", 激活：" + str(lcfg['activation']))
            self.cnnModel.appendRow(item)

    def updateMlpListView(self):
        self.mlpModel.clear()
        for lcfg in self.config['mlp']['dense_layers']:
            item = QtGui.QStandardItem("Dense: " + str(lcfg['units']) + "  激活函数: " + lcfg['activation'])
            self.mlpModel.appendRow(item)
    ## MLP
    def addMLP(self):
        print("addMLP")
        unit = self.mlpUnitEdit.text()
        unit = int(unit)
        if unit <= 0:
            self.addLogOutput("unit must be greater than 0")
            return
        activation = self.mlpActivationComboBox.currentText()
        self.config['mlp']['dense_layers'].append({'units': unit, 'activation': activation})
        self.updateMlpListView()
        self.curMlpIndex = None

    def editMLP(self, index):
        print("editMLP")
        self.curMlpIndex = index.row()
        print(self.curMlpIndex)
        self.mlpUnitEdit.setText(str(self.config['mlp']['dense_layers'][index.row()]['units']))
        self.mlpActivationComboBox.setCurrentText(self.config['mlp']['dense_layers'][index.row()]['activation'])
        return True

    def removeMLP(self):
        print("removeMLP")
        if self.curMlpIndex is None:
            self.addLogOutput("Please select a layer to remove")
            return
        self.config['mlp']['dense_layers'].pop(self.curMlpIndex)
        self.updateMlpListView()
        self.curMlpIndex = None

    def confirmEditMLP(self):
        print("confirmEditMLP")
        if self.curMlpIndex is None:
            self.addLogOutput("Please select a layer to edit")
            return
        unit = self.mlpUnitEdit.text()
        unit = int(unit)
        if unit <= 0:
            self.addLogOutput("unit must be greater than 0")
            return
        self.config['mlp']['dense_layers'][self.curMlpIndex]['units'] = unit
        self.config['mlp']['dense_layers'][self.curMlpIndex]['activation'] = self.mlpActivationComboBox.currentText()
        self.updateMlpListView()
        self.curMlpIndex = None

    ## CNN
    def addCNN(self):
        print("addCNN")
        filters = self.cnnFiltersEdit.text()
        filters = int(filters)
        kernel_size_x = self.cnnKernelSizeXEdit.text()
        kernel_size_x = int(kernel_size_x)
        kernel_size_y = self.cnnKernelSizeYEdit.text()
        kernel_size_y = int(kernel_size_y)
        if filters <= 0 or kernel_size_x <= 0 or kernel_size_y <= 0:
            self.addLogOutput("filters and kernel size must be greater than 0")
            return
        activation = self.cnnActivationComboBox.currentText()
        layerCfg = {"type": "conv2d", 'filters': filters, 'kernel_size': (kernel_size_x, kernel_size_y), 'activation': activation}
        self.config['cnn']['conv_layers'].append({'filters': filters, 'kernel_size': (kernel_size_x, kernel_size_y), 'activation': activation})
        self.updateCnnListView()
        self.curCnnIndex = None

    def editCNN(self, index):
        print("editCNN")
        self.curCnnIndex = index.row()
        print(self.curCnnIndex)
        self.cnnFiltersEdit.setText(str(self.config['cnn']['conv_layers'][index.row()]['filters']))
        self.cnnKernelSizeXEdit.setText(str(self.config['cnn']['conv_layers'][index.row()]['kernel_size'][0]))
        self.cnnKernelSizeYEdit.setText(str(self.config['cnn']['conv_layers'][index.row()]['kernel_size'][1]))
        self.cnnActivationComboBox.setCurrentText(self.config['cnn']['conv_layers'][index.row()]['activation'])
        return True
        
    def removeCNN(self):
        print("removeCNN")
        if self.curCnnIndex is None:
            self.addLogOutput("Please select a layer to remove")
            return
        self.config['cnn']['conv_layers'].pop(self.curCnnIndex)
        self.updateCnnListView()
        self.curCnnIndex = None
        
    def confirmEditCNN(self):
        print("confirmEditCNN")
        if self.curCnnIndex is None:
            self.addLogOutput("Please select a layer to edit")
            return
        filters = self.cnnFiltersEdit.text()
        filters = int(filters)
        kernel_size_x = self.cnnKernelSizeXEdit.text()
        kernel_size_x = int(kernel_size_x)
        kernel_size_y = self.cnnKernelSizeYEdit.text()
        kernel_size_y = int(kernel_size_y)
        if filters <= 0 or kernel_size_x <= 0 or kernel_size_y <= 0:
            self.addLogOutput("filters and kernel size must be greater than 0")
            return
        activation = self.cnnActivationComboBox.currentText()
        self.config['cnn']['conv_layers'][self.curCnnIndex]['filters'] = filters
        self.config['cnn']['conv_layers'][self.curCnnIndex]['kernel_size'] = (kernel_size_x, kernel_size_y)
        self.config['cnn']['conv_layers'][self.curCnnIndex]['activation'] = activation
        self.updateCnnListView()
        self.curCnnIndex = None
        
    ## config Type
    def setType(self, type):
        self.config['type'] = type

    def updateType(self):
        if self.config['type'] == "MLP":
            self.typeRadioButtonMLP.setChecked(True)
        else:
            self.typeRadioButtonCNN.setChecked(True)

    ## config Optimizer
    def setOptimizer(self, optimizer):
        self.config['optimizer'] = optimizer

    def updateOptimizer(self):
        if self.config['optimizer'] == "SGD":
            self.optimizerRadioButtonSGD.setChecked(True)
        else:
            self.optimizerRadioButtonAdam.setChecked(True)

    ## config learning rate
    def setLearningRate(self, learning_rate):
        learning_rate = float(learning_rate)
        self.config['learning_rate'] = learning_rate

    def updateLearningRate(self):
        self.learnRateEdit.setText(str(self.config['learning_rate']))

    # config framework
    def setFramework(self, framework):
        self.config['framework'] = framework

    def updateFramework(self):
        if self.config['framework'] == "Tensorflow":
            self.frameworkRadioButtonTensorflow.setChecked(True)
        else:
            self.frameworkRadioButtonKeras.setChecked(True)

    ## config max_epocs
    def setMaxIter(self, max_iter):
        max_iter = int(max_iter)
        self.config['max_iter'] = max_iter
        
    def updateMaxIter(self):
        self.maxEpocsEdit.setText(str(self.config['max_iter']))

    ## config checkpoint
    def setCheckpoint(self, checkpoint):
        checkpoint = int(checkpoint)
        self.config['checkpoint'] = checkpoint

    def updateCheckpoint(self):
        self.autoSaveEpocsEdit.setText(str(self.config['checkpoint']))

    def createRadioButton(self, text):
        radioButton = QtWidgets.QRadioButton(text)
        radioButton.toggled.connect(self.onRadioButtonToggled)
        return radioButton
    
    def onRadioButtonToggled(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            print(radioButton.text())
            self.setType(radioButton.text())

    ## show image
    def showImage(self):
        if self.checkThread() is None:
            self.data = self.trainThread.data
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

    ## log output
    def addLogOutput(self, output):
        self.outputTextBrowser.append(output)
        self.outputTextBrowser.update()
        vscrollBar = self.outputTextBrowser.verticalScrollBar()
        vscrollBar.setValue(vscrollBar.maximum())
    
    ## thread control
    def initThread(self):
        if self.logcallback == None:
            self.logcallback = MyLoggerCallback()
            self.logcallback.setSender(self)
            self.trainThread = TrainThread()
            self.trainThread.setLogCallBack(self.logcallback)
            self.trainThread.setLogOutput(self)
            self.trainThread.start()

    def checkThread(self):
        if self.trainThread.threadIsIdle():
            return None
        msg = self.trainThread.getStatusMessage()
        return msg

    ## command control        
    def doTrain(self):
        self.initThread()
        msg = self.checkThread()
        if msg is not None:
            self.addLogOutput(msg)
            return
        self.trainThread.setConfig(self.config)
        self.trainThread.setContinue(False)
        self.trainThread.setStatus("Training")
        self.outputTextBrowser.clear()
        self.addLogOutput("start training thread")
        self.outputTextBrowser.update()
    
    def doContinue(self):
        self.initThread()
        msg = self.checkThread()
        if msg is not None:
            self.addLogOutput(msg)
            return
        self.trainThread.setConfig(self.config)
        self.trainThread.setContinue(True)
        self.trainThread.setStatus("Training")
        self.outputTextBrowser.clear()
        self.addLogOutput("Start continue training thread")
        self.outputTextBrowser.update()

    def doTest(self):
        self.initThread()
        msg = self.checkThread()
        if msg is not None:
            self.addLogOutput(msg)
            return
        self.trainThread.setStatus("Testing")
        self.outputTextBrowser.clear()
        self.addLogOutput("Start testing thread")
        self.outputTextBrowser.update()

    ## train precision
    def setTrainPrecision(self, precision):
        self.config['precision'] = precision

    def updateTrainPrecision(self):
        try:
            showString = "{:.4}".format(self.config['precision']*100)
            self.trainPrecisionLabel.setText(showString)
        except:
            pass
        
    ## test precision
    def setTestPrecision(self, precision):
        self.config['precision'] = precision

    def updateTestPrecision(self):
        try:
            showString = "{:.4}".format(self.config['precision']*100)
            self.testPrecisionLabel.setText(showString)
        except:
            pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = MainDialog()
    Dialog.show()
    sys.exit(app.exec_())
