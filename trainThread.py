#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
from fer2013Dataset import FER2013Dataset, pixel_to_image
from train_tensorflow import train, test
from PyQt5.QtCore import QThread


class TrainThread(QThread):
    THREAD_STATUS = ["Idle", "Training", "Testing", "DataLoading"]
    

    def __init__(self):
        super(QThread, self).__init__()
        self.callback = None
        self.output = None
        self.is_continue = False
        self.config = None
        self.data = None
        self.thread_status = 0

    def setStatus(self, status: str):
        statusidx = self.THREAD_STATUS.index(status)
        if statusidx == -1:
            return
        self.thread_status = statusidx

    def setLogCallBack(self, callback):
        self.callback = callback

    def setLogOutput(self, output):
        self.output = output

    def threadIsIdle(self):
        return self.thread_status == 0
    
    def getStatusMessage(self):
        statusStr = self.THREAD_STATUS[self.thread_status]
        # "Training is already started"
        if self.threadIsIdle():
            return_message = "Thread is Idle."
        return_message = f"{statusStr} is already started, please wait"
        return return_message

    def setContinue(self, is_continue: bool):
        self.is_continue = is_continue

    def setConfig(self, config):
        if not self.threadIsIdle():
            return self.getStatusMessage()
        self.config = config
        return "config set successfully"

    def run(self):
        time.sleep(3)
        self.setStatus("DataLoading")
        while True:
            if self.threadIsIdle():
                self.output.addLogOutput(self.getStatusMessage())
                time.sleep(1)
                continue
            if self.thread_status == 3 and self.data is None:
                self.output.addLogOutput("Start loading data")
                data = FER2013Dataset(sender=self.output)
                self.data = data
                self.output.addLogOutput("Data loaded")
                self.setStatus("Idle")
            elif self.thread_status == 1:
                self.output.addLogOutput("Start training")
                train(self.callback, self.data, self.is_continue, self.config)
                self.output.addLogOutput("Training done")
                self.setStatus("Idle")
            elif self.thread_status == 2:
                self.output.addLogOutput("Start testing")
                test(self.callback, self.data, self.config)
                self.output.addLogOutput("Testing done")
                self.setStatus("Idle")
            time.sleep(1)
