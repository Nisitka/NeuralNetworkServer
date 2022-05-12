import os
import time
import copy

import numpy

import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
from tensorflow import keras
import keras.layers
from keras.datasets import mnist
from keras import models

from matplotlib import pyplot

from PyQt5 import QtCore, QtGui, QtWidgets
from tkinter import *

import socket as Socket
import codecs

from multiprocessing import Process
import threading

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
test_labels = tensorflow.keras.utils.to_categorical(test_labels)

condInputData = threading.Condition()
coundOutputData = threading.Condition()

# ---------------------------------------------------Сервер-------------------------------------------------------------
class Server:
    def __init__(self):
        self.address = "localhost"
        self.port = 2323
        self.dataPackageSize = 2048 * 1000 * 1000

        self.socket = Socket.socket()
        self.socket.bind((self.address, self.port))
        self.socket.listen(1)  # максимальное кол-во подключений

        self.numPacked = 0
        global maxPacked
        global dataNetInput
        maxPacked = 20

        self.stop = False

    def start(self):
        self.connection, self.clientAddress = self.socket.accept()
        print(f"{self.clientAddress} has connected")

        arrFull = []
        while not self.stop:
            data = self.connection.recv(self.dataPackageSize)
            if not data:
                break
            self.numPacked += 1

            # преобразование входного битового значения в одномерный массив
            arr = [int(data[i:i+2], 16) for i in range(0, len(data), 2)]
            # print(arr)
            arrFull += arr

            if self.numPacked != maxPacked:
                dataOUT = "Server get packed!"

            else:
                dataOUT = "Server get data!"
                self.numPacked = 0

                with condInputData:
                    # инициализвция входного значения для сети
                    global dataNetInput
                    dataNetInput = arrFull
                    arrFull = []

                    # вызов нейронной сети
                    condInputData.notify()

            dataOUT = codecs.encode(dataOUT, 'UTF-8')
            self.connection.sendall(dataOUT)

    def dataExport(self, data_):
        data_ = codecs.encode(str(data_), 'UTF-8')
        self.connection.sendall(data_)
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------Интерфейс и нейронная сеть-------------------------------------------------------------------
class Ui_Form(object):
    def __init__(self):
        self.Form = QtWidgets.QWidget()
        self.setupUi()

        # импорт сети (надо будет добавить искдючение)
        self.net = keras.models.load_model('net1')

        # создание и запуск потока для постоянной работы сети
        self.threadNet = threading.Thread(target=self.startNeuronsNetwork)
        self.threadNet.start()

        # инициализация и запуск сервера
        self.server = Server()
        self.threadServer = threading.Thread(target=self.startServer)
        self.threadServer.start()

    def startNeuronsNetwork(self):
        self.textInfo.append("The neural network is running")

        while True:
            with condInputData:
                condInputData.wait()

                self.processingNet()

    def startServer(self):
        self.server.start()

    def restartServer(self):
        print(2)
        self.StackedLabelImage.setCurrentIndex(1)

    def processingNet(self):
        self.textInfo.append("Data received from the client: " +
                             str(dataNetInput)[0:25] + " ... " +
                             str(dataNetInput)[len(str(dataNetInput))-25:]
                             )

        self.npArr = np.array(dataNetInput)
        w = 600
        l = 500
        self.npArr = np.reshape(self.npArr, (w, l, 3))

        self.img = numpy2pil(self.npArr)

        self.img.save("inputData.jpg")

        self.listLabelPix[self.numelGetImage % self.maxImage].setPixmap(QtGui.QPixmap("inputData.jpg"))
        self.StackedLabelImage.addWidget(self.listLabelPix[self.numelGetImage % self.maxImage])
        self.StackedLabelImage.setCurrentIndex(self.numelGetImage % self.maxImage)

        # работа сети и отправка результата
        out = self.net.evaluate(test_images, test_labels)
        self.server.dataExport(out)

        self.numelGetImage += 1

    def setTextlabel(self):
        print(self.net)
        resul = self.net.evaluate(test_images, test_labels)
        data = test_images
        print(type(test_images))

        outNet = self.net.predict(test_images[:1])[0]
        # print(max(outNet))
        maxValue = max(outNet)
        print(range(outNet.size))

        for i in range(outNet.size):
            if maxValue == outNet[i]:
                print(i)
        self.label.setText("Результаты тестовой выборки: " +
                           str(resul[0]) + " " +
                           str(resul[1])
                           )

    def setupUi(self):
        self.Form.setWindowTitle("Server")
        self.Form.setStyleSheet('background-color: rgb(49,54,72)')
        # self.Form.setStyleSheet("QPushButton { background-color: yellow }") #цвет всех кнопок

        # кнопка рестарта сервера
        self.pushButton = QtWidgets.QPushButton(self.Form)
        self.pushButton.setText("Restart server")
        self.pushButton.clicked.connect(self.restartServer)
        self.pushButton.setStyleSheet('''
                    QPushButton {
                        background-color: rgb(124,128,219); color: rgb(40,40,40);
                    }
                    QPushButton:pressed {
                        background-color : rgb(94,98,189); color: rgb(10,10,10);
                    }
                ''')

        # многострочное текстовое поле
        self.textInfo = QtWidgets.QTextEdit(self.Form)
        self.textInfo.setObjectName("infoTextDisp")
        self.textInfo.setReadOnly(True)
        self.textInfo.setMinimumSize(450, 600)

        self.textInfo.setStyleSheet('background-color: rgb(89,94,112); color: rgb(200,200,200)')

        self.HBoxMain = QtWidgets.QHBoxLayout(self.Form)

        self.VBoxText = QtWidgets.QVBoxLayout(self.Form)
        self.VBoxText.addWidget(self.textInfo)
        self.VBoxText.addWidget(self.pushButton)

        # лэйбел с картинкой входной информации

        self.indexInDataImage = 0
        self.StackedLabelImage = QtWidgets.QStackedWidget(self.Form)

        # self.LabelPixels = QtWidgets.QLabel(self.StackedLabelImage)
        self.listLabelPix = []
        self.maxImage = 5
        self.numelGetImage = 0
        for i in range(self.maxImage):
            self.listLabelPix.append(QtWidgets.QLabel(self.StackedLabelImage))

        self.ImagesInterface = QtWidgets.QVBoxLayout(self.Form)
        self.pushButtonBox = QtWidgets.QHBoxLayout(self.Form)

        self.pushButtonNext = QtWidgets.QPushButton(self.Form)
        self.pushButtonNext.clicked.connect(self.nextImage)
        self.pushButtonNext.setText(">")
        self.pushButtonNext.setStyleSheet('''
                    QPushButton {
                        background-color: rgb(124,128,219); color: rgb(40,40,40);
                    }
                    QPushButton:pressed {
                        background-color : rgb(94,98,189); color: rgb(10,10,10);
                    }
                ''')

        self.pushButtonPast = QtWidgets.QPushButton(self.Form)
        self.pushButtonPast.clicked.connect(self.pastImage)
        self.pushButtonPast.setText("<")
        self.pushButtonPast.setStyleSheet('''
                    QPushButton {
                        background-color: rgb(124,128,219); color: rgb(40,40,40);
                    }
                    QPushButton:pressed {
                        background-color : rgb(94,98,189); color: rgb(10,10,10);
                    }
                ''')

        self.pushButtonBox.addWidget(self.pushButtonPast)
        self.pushButtonBox.addWidget(self.pushButtonNext)

        self.ImagesInterface.addWidget(self.StackedLabelImage)
        self.ImagesInterface.addLayout(self.pushButtonBox)

        # self.HBoxMain.addWidget(self.StackedLabelImage)
        self.HBoxMain.addLayout(self.ImagesInterface)
        self.HBoxMain.addLayout(self.VBoxText)

        self.Form.setLayout(self.HBoxMain)

        QtCore.QMetaObject.connectSlotsByName(self.Form)

    def nextImage(self):
        if self.indexInDataImage < self.maxImage-1 and self.indexInDataImage < self.numelGetImage-1:
            print("+")
            self.indexInDataImage += 1
            self.StackedLabelImage.setCurrentIndex(self.indexInDataImage)

    def pastImage(self):
        if self.indexInDataImage > 0:
            print("-")
            self.indexInDataImage -= 1
            self.StackedLabelImage.setCurrentIndex(self.indexInDataImage)

    def exec(self):
        self.Form.show()
# ----------------------------------------------------------------------------------------------------------------------

import os.path
import numpy as np
from PIL import Image


def pil2numpy(img: Image = None) -> np.ndarray:
    if img is None:
        img = Image.open('Nisitka.jpg')

        np_array = np.asarray(img)
        return np_array


def numpy2pil(np_array: np.ndarray) -> Image:
    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    uiProgram = Ui_Form()
    uiProgram.exec()

    sys.exit(app.exec_())




