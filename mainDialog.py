# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1337, 994)
        self.listView = QtWidgets.QListView(Dialog)
        self.listView.setGeometry(QtCore.QRect(40, 40, 251, 311))
        self.listView.setObjectName("listView")
        self.listView_2 = QtWidgets.QListView(Dialog)
        self.listView_2.setGeometry(QtCore.QRect(340, 41, 256, 311))
        self.listView_2.setObjectName("listView_2")
        self.editMlpButton = QtWidgets.QPushButton(Dialog)
        self.editMlpButton.setGeometry(QtCore.QRect(150, 360, 100, 32))
        self.editMlpButton.setObjectName("editMlpButton")
        self.editCNNButton = QtWidgets.QPushButton(Dialog)
        self.editCNNButton.setGeometry(QtCore.QRect(480, 360, 100, 32))
        self.editCNNButton.setObjectName("editCNNButton")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(640, 40, 271, 311))
        self.graphicsView.setObjectName("graphicsView")
        self.showImgButton = QtWidgets.QPushButton(Dialog)
        self.showImgButton.setGeometry(QtCore.QRect(800, 360, 100, 32))
        self.showImgButton.setObjectName("showImgButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "AI综合实践"))
        self.editMlpButton.setText(_translate("Dialog", "编辑MLP"))
        self.editCNNButton.setText(_translate("Dialog", "编辑CNN"))
        self.showImgButton.setText(_translate("Dialog", "显示图片"))
