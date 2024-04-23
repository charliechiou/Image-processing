from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from PIL import Image
from torchsummary import summary
from torchvision import datasets
import os
import sys
import main_gui
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import random

class animalDataset(Dataset):
    def __init__(self,img_dir,transforms = None):
        self.img_dir = img_dir
        self.transform = transforms
        self.subfolders = [f for f in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir,f))]
        self.img_labels = []
        self.img = []

        for subfolder in self.subfolders:
            files = os.listdir(os.path.join(self.img_dir,subfolder))
            subfolder_name = os.path.join(img_dir,subfolder)
            for f in files:
                self.img_labels.append(subfolder)
                img_name = os.path.join(subfolder_name,f)
                self.img.append(img_name)

    def __len__(self):
        count = 0
        for subfolder in self.subfolders:
            files = os.listdir(os.path.join(self.img_dir,subfolder))
            count += len(files)
        return count
    
    def __getitem__(self,idx):
        image = read_image(self.img[idx])
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        
        return image,label

class DoodleWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.pixmap = QPixmap(564,349)
        self.pixmap.fill(Qt.black)

        self.drawing = False
        self.last_point = QPoint()
        self.acc_show = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect().topLeft(), self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.acc_show == False:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(Qt.white, 25, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            painter.end()

            self.update()
            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def reset_canvas(self):
        self.pixmap = QPixmap(564,349)
        self.pixmap.fill(Qt.black)
        self.acc_show = False
        self.update()

    def get_img(self):
        qimage = self.pixmap.toImage()
        qimage.save('draw.png')

    def show_data(self):
        self.acc_show = True
        self.pixmap = QPixmap(".\\runs\\num\data_resize.png").scaled(564, 309, Qt.KeepAspectRatio)
        self.update()

class MainWindows(QFrame, main_gui.Ui_hw2):
    def __init__(self):
        super().__init__()
        self.windows = []
        self.setupUi(self)

        #模型設定
        self.transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]
        )


        #VGG19模型
        self.vgg19 = torch.load(
            ".\model\\num\epochs_80.pt")
        
        self.vgg19.to('cpu')
        self.vgg19.eval()
        self.classes = ('0', '1', '2', '3',
                        '4', '5', '6', '7', '8', '9')

        #模型設定
        self.transform_animal = transforms.Compose(
            [transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])
        
        #resNet模型
        
        #未使用RandomErasing
        # self.resNet = torch.load(
        #     "C:\chiou\opencv_class\hw2\model\\animal\\30epoch_128\\30epoch_128.pt"
        # )

        #使用RandomErasing
        self.resNet = torch.load(
            ".\model\\animal\\30epoch_128_re\\30epoch_128_re.pt"
        )

        self.resNet.to('cpu')
        self.resNet.eval()

        #load image
        self.load_img_btn.clicked.connect(self.load_img_btn_clicked)

        #question 1
        self.q11_btn.clicked.connect(self.q11)
        self.q12_btn.clicked.connect(self.q12)

        #question 2
        self.q2_btn.clicked.connect(self.q2)

        #question 3
        self.q31_btn.clicked.connect(self.q31)
        self.q32_btn.clicked.connect(self.q32)

        #question 4
        self.painting = DoodleWidget()
        self.q4_display_box.addWidget(self.painting)

        self.q41_btn.clicked.connect(self.q41)
        self.q42_btn.clicked.connect(self.q42)
        self.q43_btn.clicked.connect(self.q43)
        self.q44_btn.clicked.connect(self.q44)

        #question 5
        self.q5_load_img_btn.clicked.connect(self.q5_load_img)
        self.q51_btn.clicked.connect(self.q51)
        self.q52_btn.clicked.connect(self.q52)
        self.q53_btn.clicked.connect(self.q53)
        self.q54_btn.clicked.connect(self.q54)

    #讀取相片
    def load_img_btn_clicked(self, index):
        try:
            file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
            self.f_name = file_name[0]
            f_name = self.f_name.split('/')
            self.load_img_display.setText(f_name[-1])
            print(f'"{f_name[-1]}" is now opening ...')
        except:
            print("No file opened!")

    def HoughCircle(self):
        #前處理
        img = cv2.imread(self.f_name)
        img_pre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_pre = cv2.GaussianBlur(img_pre,(5,5),0)

        #Hough Circle Transform
        circles = cv2.HoughCircles(img_pre,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=20,maxRadius=50)
        circles = np.uint16(np.around(circles))
        return img,circles

    def q11(self):

        img,circles = self.HoughCircle()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.title('img_src')
        plt.imshow(img)
        plt.show()
        center = np.zeros(img.shape)
        
        #圈出每個圓
        for i in circles[0]:
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(center,(i[0],i[1]),2,(255,255,255),3)

        #顯示最終圖像
        plt.figure()
        plt.title('img_process')
        plt.imshow(img)
        plt.show()
        
        plt.figure()
        plt.title('Circle_center')
        plt.imshow(center)
        plt.show()
    
    def q12(self):
        _,circles = self.HoughCircle()
        self.q1_display.setText(f'There are {circles.shape[1]} coins in the image.')

    def hist_eq(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hist, bins = np.histogram(img,bins=256,range=(0,256))
        cdf = np.zeros(256)
        lut = np.zeros(256)
        sum = 0

        non_zero = np.nonzero(hist)
        min_id = non_zero[0][0]
        max_id = non_zero[0][::-1][0]
        
        for i in range(0,256):
            sum += hist[i]
            cdf[i] = sum
        
        for i in range(0,256):
            lut[i] = round((cdf[i]-cdf[min_id])/(cdf[max_id]-cdf[min_id])*255)

        img_hist = lut[img].astype(np.uint8)

        return img_hist
    
    def q2(self):
        img = cv2.imread(self.f_name)
        img_pre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_pre)
        img_manual = self.hist_eq(img)

        hist_size = 255
        hist_range = [0,255]
        
        hist_img = cv2.calcHist([img], [0], None, [hist_size], hist_range)
        hist_img = np.reshape(hist_img,-1)
        hist_img_eq = cv2.calcHist([img_eq], [0], None, [hist_size], hist_range)
        hist_img_eq = np.reshape(hist_img_eq,-1)
        hist_img_manual = cv2.calcHist([img_manual], [0], None, [hist_size], hist_range)
        hist_img_manual = np.reshape(hist_img_manual,-1)

        plt.figure(figsize=(16,9))
    
        plt.subplot(2,3,1)
        plt.title('Original Image')
        plt.axis('off')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)

        plt.subplot(2,3,2)
        plt.title('Equalized with OpenCV')
        plt.axis('off')
        img_eq = cv2.cvtColor(img_eq,cv2.COLOR_BGR2RGB)
        plt.imshow(img_eq)

        plt.subplot(2,3,3)
        plt.title('Equalized with Manual')
        plt.axis('off')
        img_manual = cv2.cvtColor(img_manual,cv2.COLOR_BGR2RGB)
        plt.imshow(img_manual)

        plt.subplot(2,3,4)
        plt.title('Histogram of Original')
        plt.xlabel('Gray Scale')
        plt.ylabel('Frequenct')
        plt.bar(range(1,hist_size + 1),hist_img)

        plt.subplot(2,3,5)
        plt.title('Histogram of Equalized(OpenCV)')
        plt.xlabel('Gray Scale')
        plt.ylabel('Frequenct')
        plt.bar(range(1,hist_size + 1),hist_img_eq)

        plt.subplot(2,3,6)
        plt.title('Histogram of Equalized(Manual)')
        plt.xlabel('Gray Scale')
        plt.ylabel('Frequenct')
        plt.bar(range(1,hist_size + 1),hist_img_manual)

        plt.tight_layout()
        plt.show()

    def dia(self,img):
        SE_nums = 3
        SE = np.ones(shape=(SE_nums,SE_nums),dtype=np.uint8)
        m,n = img.shape
        a = int((SE_nums-1)/2)
        img_dia = np.zeros((m-2*a,n-2*a))
        for i in range(a,m-a):
            for j in range(a,n-a):
                temp = img[i-1:i+2,j-1:j+2]
                pd = temp * SE
                img_dia[i-1,j-1] = np.max(pd)
        return img_dia

    def ero(self,img):
        SE_nums = 3
        SE = np.ones(shape=(SE_nums,SE_nums),dtype=np.uint8)
        m,n = img.shape
        a = int((SE_nums-1)/2)
        img_ero = np.zeros((m-2*a,n-2*a))
        for i in range(a,m-a):
            for j in range(a,n-a):
                temp = img[i-1:i+2,j-1:j+2]
                pd = temp * SE
                img_ero[i-1,j-1] = np.min(pd)
        return img_ero

    def q31(self):
        img = cv2.imread(self.f_name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,img_bi = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        img_clo = self.ero(self.dia(img_bi))
        cv2.imshow('ori',img)
        cv2.imshow('close',img_clo)

    def q32(self):
        img = cv2.imread(self.f_name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,img_bi = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        img_open = self.dia(self.ero(img_bi))
        cv2.imshow('ori',img)
        cv2.imshow('open',img_open)

    def q41(self):
        os.system('cls')
        summary(self.vgg19, (1, 32, 32),device='cpu')
    
    def q42(self):
        self.painting.show_data()

    def q43(self):
        self.painting.get_img()
        img_class = Image.open('./draw.png').crop((0,0,550,350))
        # img_crop.show()
        img_tensor = self.transform(img_class)
        with torch.no_grad():
            outputs = self.vgg19(img_tensor.unsqueeze(0))
            _, predicted = torch.max(outputs, 1)

        print(self.classes[predicted])

        #套用sofymax並顯示結果
        s = nn.Softmax(dim=1)
        sig = s(outputs)
        pro_show = sig.detach().numpy()

        #顯示於gui中
        self.q4_num.setText(f'Predict = {self.classes[predicted]}')
        plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                pro_show[0], tick_label=self.classes)
        plt.ylim((0, 1))
        plt.show()

    def q44(self):
        self.painting.reset_canvas()

    def q5_load_img(self):
        try:
            q5_img = QFileDialog.getOpenFileName(self, 'open file', '.')
            self.q5_img = q5_img[0]
            print(self.q5_img)
        except:
            print("No file opened!")

    def q51(self):
        transform_dst = transforms.Compose([transforms.Resize((224,224),antialias=True)])
        test_dataset = animalDataset('.\Dataset_OpenCvDl_Hw2_Q5\dataset\inference_dataset',transforms=transform_dst)
        cat = []
        dog = []
        for i in range(len(test_dataset)):
            img = test_dataset[i]
            if img[1] == 'Cat':
                cat.append(i)
            if img[1] == 'Dog':
                dog.append(i)

        cat_tensor = test_dataset[random.choice(cat)]
        cat = cat_tensor[0].permute(1,2,0).numpy()

        dog_tensor = test_dataset[random.choice(dog)]
        dog = dog_tensor[0].permute(1,2,0).numpy()

        plt.figure()
        plt.subplot(121)
        plt.imshow(cat)
        plt.title(cat_tensor[1])
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(dog)
        plt.title(dog_tensor[1])
        plt.axis('off')

        plt.show()


    def q52(self):
        os.system('cls')
        resnet50 = torchvision.models.resnet50()
        resnet50.fc = nn.Sequential(
            nn.Linear(2048,1),
            nn.Sigmoid()
        )

        summary(resnet50, (3, 64, 64),device='cpu')

    def q53(self):
        acc = cv2.imread(".\\runs\Accuracy Comparison.png")
        cv2.imshow('acc',acc)
    def q54(self):
        #將照片顯示在gui中
        ori_img = cv2.imread(self.q5_img)
        img = cv2.resize(ori_img, (224, 224))
        height, width, channel = img.shape
        bytePerline = channel * width
        qImg = QImage(img, width, height, bytePerline,
                          QImage.Format_BGR888)
        self.q5_display.setPixmap(QPixmap.fromImage(qImg))
        #辨識照片
        img_class = Image.open(self.q5_img)
        img_tensor = self.transform_animal(img_class)

        sample = torch.unsqueeze(img_tensor,dim=0)


        with torch.no_grad():
            outputs = self.resNet(sample)
            outputs = torch.sigmoid(outputs)

        if outputs > 0.5 :
            self.q5_show.setText('Prediction: Dog')
            print('dog')
        if outputs < 0.5 :
            self.q5_show.setText('Prediction: Cat')
            print('cat')



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_windows = MainWindows()

    main_windows.show()
    sys.exit(app.exec_())
