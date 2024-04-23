from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from torchsummary import summary
import os
import sys
import main_gui


class MainWindows(QFrame, main_gui.Ui_hw1):
    def __init__(self):
        super().__init__()
        self.windows = []
        self.setupUi(self)

        #設定各題按鈕
        self.load_img_btn1.clicked.connect(self.load_img_btn1_clicked)
        self.load_img_btn2.clicked.connect(self.load_img_btn2_clicked)

        self.btn_11.clicked.connect(self.question_11)
        self.btn_12.clicked.connect(self.question_12)
        self.btn_13.clicked.connect(self.question_13)

        self.btn_21.clicked.connect(self.question_21)
        self.btn_22.clicked.connect(self.question_22)
        self.btn_23.clicked.connect(self.question_23)

        self.btn_31.clicked.connect(self.question_31)
        self.btn_32.clicked.connect(self.question_32)
        self.btn_33.clicked.connect(self.question_33)
        self.btn_34.clicked.connect(self.question_34)

        self.btn_4.clicked.connect(self.question_4)

        self.btn_51.clicked.connect(self.question_51)
        self.btn_52.clicked.connect(self.question_52)
        self.btn_53.clicked.connect(self.question_53)
        self.load_img_5_btn.clicked.connect(self.load_img_classify)
        self.btn_54.clicked.connect(self.question_54)

        #VGG19設定
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.vgg19 = torch.load(
            ".\model\epoch_120\model.pt")
        self.vgg19.to('cpu')
        self.vgg19.eval()
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #開啟相片
    def load_img_btn1_clicked(self, index):
        try:
            file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
            self.f_name1 = file_name[0]
            f_name = self.f_name1.split('/')
            self.load_img1.setText(f_name[-1])
            print(f'"{f_name[-1]}" is now opening ...')
        except:
            print("No file opened!")

    def load_img_btn2_clicked(self, index):
        try:
            file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
            self.f_name2 = file_name[0]
            f_name = self.f_name2.split('/')
            self.load_img2.setText(f_name[-1])
            print(f'"{f_name[-1]}" is now opening ...')
        except:
            print("No file opened!")

    def load_img_classify(self, index):
        try:
            file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
            self.f_class = file_name[0]
        except:
            print("No file opened!")

    def question_11(self):
        ori_img = cv2.imread(self.f_name1)
        ori_img_show = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        size = ori_img.shape

        plt.subplot(2, 3, 2)
        plt.imshow(ori_img_show)
        plt.title("original image")

        #分割b r g channel
        b, g, r = cv2.split(ori_img)
        zero = np.zeros((size[0], size[1]), dtype="uint8")

        #分別重組三channel
        b = cv2.merge([b, zero, zero])
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

        g = cv2.merge([zero, g, zero])
        g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)

        r = cv2.merge([zero, zero, r])
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)

        #顯示
        plt.subplot(2, 3, 4)
        plt.imshow(b)
        plt.title("B channel")

        plt.subplot(2, 3, 5)
        plt.imshow(g)
        plt.title("G channel")

        plt.subplot(2, 3, 6)
        plt.imshow(r)
        plt.title("R channel")

        plt.show()

    def question_12(self):
        ori_img = cv2.imread(self.f_name1)

        #分割三channel
        b, g, r = cv2.split(ori_img)

        img_cv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

        #合併
        img_ave = (b+g+r)/3

        #後處理
        img_ave = img_ave.astype('uint8')
        img_cv = img_cv.astype('uint8')

        #合併圖片並顯示
        img_show = np.concatenate((img_cv, img_ave), axis=1)
        cv2.imshow("question 1.2", img_show)

    def question_13(self):
        ori_img = cv2.imread(self.f_name1)
        hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)

        #設定上下限
        lowerBound = (10, 25, 25)
        upperBound = (85, 255, 255)

        #建立遮罩
        mask = cv2.inRange(hsv_img, lowerBound, upperBound)

        #顯示遮罩並套用
        mask_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        remove_img = cv2.bitwise_not(mask_show, ori_img, mask)

        #合併結果並顯示
        img_show = np.concatenate((mask_show, remove_img), axis=1)
        cv2.imshow('question 1.3', img_show)

    def question_21(self):
        self.ori_img = cv2.imread(self.f_name1)
        cv2.imshow("question 2.1", self.ori_img)
        cv2.createTrackbar("m", "question 2.1", 1, 5, self.gaussian)

    #建立gaussian filter
    def gaussian(self, val):
        output = cv2.GaussianBlur(self.ori_img, (2*val+1, 2*val+1), 0)
        cv2.imshow("question 2.1", output)

    def question_22(self):
        self.ori_img = cv2.imread(self.f_name1)
        cv2.imshow("question 2.2", self.ori_img)
        cv2.createTrackbar("m", "question 2.2", 1, 5, self.bilateral)

    #建立bilateral filter
    def bilateral(self, val):
        output = cv2.bilateralFilter(self.ori_img, 2*val+1, 90, 90)
        cv2.imshow("question 2.2", output)

    def question_23(self):
        self.ori_img = cv2.imread(self.f_name1)
        cv2.imshow("question 2.3", self.ori_img)
        cv2.createTrackbar("m", "question 2.3", 1, 5, self.median)

    #建立median filter
    def median(self, val):
        output = cv2.medianBlur(self.ori_img, 2*val+1)
        cv2.imshow("question 2.3", output)

    def question_31(self):
        ori_img = cv2.imread(self.f_name1)
        gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

        #套用gaussian filter前處理
        gau_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        #套用sobel filter
        x = self.sobel_X(gau_img)
        cv2.imshow("question 3.1", x/256)

    def question_32(self):
        ori_img = cv2.imread(self.f_name1)
        gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

        #套用gaussian filter前處理
        gau_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        #套用sobel filter
        y = self.sobel_Y(gau_img)
        cv2.imshow("question 3.2", y/256)

    def sobel_X(self, x):
        #建立遮罩
        filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        size = np.shape(x)
        new_img = np.zeros((size[0]-2, size[1]-2))

        #遍歷各像素並套用遮罩
        for i in range(1, size[0]-2):
            for j in range(1, size[1]-2):
                new_img[i-1][j-1] = (x[i-1][j-1]*filter[0][0])+(x[i-1][j]*filter[0][1])+(x[i-1][j+1]*filter[0][2]) + \
                                    (x[i][j-1]*filter[1][0])+(x[i][j]*filter[1][1])+(x[i][j+1]*filter[1][2]) + \
                                    (x[i+1][j-1]*filter[2][0])+(x[i+1][j] *
                                                                filter[2][1])+(x[i+1][j+1]*filter[2][2])
        return new_img

    def sobel_Y(self, x):
        #建立遮罩
        filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        size = np.shape(x)
        new_img = np.zeros((size[0]-2, size[1]-2))

        #遍歷各像素並套用遮罩
        for i in range(1, size[0]-2):
            for j in range(1, size[1]-2):
                new_img[i-1][j-1] = (x[i-1][j-1]*filter[0][0])+(x[i-1][j]*filter[0][1])+(x[i-1][j+1]*filter[0][2]) + \
                                    (x[i][j-1]*filter[1][0])+(x[i][j]*filter[1][1])+(x[i][j+1]*filter[1][2]) + \
                                    (x[i+1][j-1]*filter[2][0])+(x[i+1][j] *
                                                                filter[2][1])+(x[i+1][j+1]*filter[2][2])
        return new_img

    def question_33(self):
        #前處理
        ori_img = cv2.imread(self.f_name1)
        gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        gau_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        #套用遮罩
        x = self.sobel_X(gau_img)
        y = self.sobel_Y(gau_img)

        #合併x y方向
        x_s = np.square(x)
        y_s = np.square(y)
        new_img = np.sqrt(x_s+y_s)
        size = np.shape(new_img)

        #設定閥值
        threshold = 200
        bi_img = np.zeros((size[0], size[1]))
        #二值化
        for i in range(size[0]):
            for j in range(size[1]):
                if new_img[i][j] > threshold:
                    bi_img[i][j] = 255
                else:
                    bi_img[i][j] = 0

        #合併並顯示
        img_show = np.concatenate((new_img/256, bi_img), axis=1)
        cv2.imshow("question 3.3", img_show)

    def question_34(self):
        #前處理
        ori_img = cv2.imread(self.f_name1)
        gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        gau_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        #套用遮罩
        x = self.sobel_X(gau_img)
        y = self.sobel_Y(gau_img)

        #合併x y方向
        x_s = np.square(x)
        y_s = np.square(y)
        new_img = np.sqrt(x_s+y_s)

        #計算各像素角度
        degree_mat = np.arctan2(y, x)
        degree_mat = degree_mat*180/np.pi

        #建立遮罩
        mask1 = cv2.inRange(degree_mat, 120, 180)
        mask2 = cv2.inRange(degree_mat, -150, -30)

        #後處理並輸出
        output1 = cv2.bitwise_and(new_img.astype('uint8'), mask1)
        output2 = cv2.bitwise_and(new_img.astype("uint8"), mask2)

        img_show = np.concatenate((output1, output2), axis=1)
        cv2.imshow("test", img_show)

    def question_4(self):
        #前處理並讀取gui數值
        ori_img = cv2.imread(self.f_name1)
        rotation = self.rotation_input.toPlainText()
        scaling = self.scaling_input.toPlainText()
        tx = self.tx_input.toPlainText()
        ty = self.ty_input.toPlainText()

        #錯誤處理
        try:
            rotation = float(rotation)
        except:
            print("The rotation input is invalid")

        try:
            scaling = float(scaling)
        except:
            print("The scaling input is invalid")

        try:
            tx = int(tx)
        except:
            print("The tx input is invalid")

        try:
            ty = int(ty)
        except:
            print("The ty input is invalid")

        #取得圖片大小及中心值
        height, width = ori_img.shape[:2]
        center_x, center_y = (width/2, height/2)

        #取得Affine matric
        M = cv2.getRotationMatrix2D((center_x, center_y), rotation, scaling)
        T = np.float32([[1, 0, tx],
                        [0, 1, ty]])

        #套用Affine matric並顯示
        img_trans = cv2.warpAffine(ori_img, T, (width, height))
        img_rotated = cv2.warpAffine(img_trans, M, (width, height))
        cv2.imshow('question 4', img_rotated)

    def question_51(self):
        #讀取照片
        img = []
        img_after = []
        allFileList = os.listdir('.\Dataset_OpenCvDl_Hw1\Q5_image\Q5_1')
        PATH = '.\Dataset_OpenCvDl_Hw1\Q5_image\Q5_1\\'

        #圖片前處理
        transform = v2.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(30)])
        fig = plt.figure()

        #顯示圖片並套用
        for i in range(9):
            img.append(Image.open(PATH+allFileList[i]))
            img_after.append(transform(img[i]))
            figs = fig.add_subplot(3, 3, i+1)
            figs.title.set_text(allFileList[i][:-4])
            plt.imshow(img_after[i])
        plt.tight_layout()
        plt.show()

    def question_52(self):
        #顯示架構
        
        summary(self.vgg19, (3, 32, 32),device='cpu')

    def question_53(self):
        #讀取並顯示訓練結果
        acc = cv2.imread('.\model\epoch_120\\accuracy.png')
        loss = cv2.imread('.\model\epoch_120\\loss.png')
        cv2.imshow('Loss', loss)
        cv2.imshow('accuracy', acc)

    def question_54(self):
        #讀取照片並顯示於gui中
        try:
            ori_img = cv2.imread(self.f_class)
            img = cv2.resize(ori_img, (128, 128))
            height, width, channel = img.shape
            bytePerline = channel * width
            qImg = QImage(img, width, height, bytePerline,
                          QImage.Format_BGR888)
            self.classify_display.setPixmap(QPixmap.fromImage(qImg))

            #以PIL讀取照片
            img_class = Image.open(self.f_class)

            #前處理
            img_tensor = self.transform(img_class)

            #丟入模型中分析
            with torch.no_grad():
                outputs = self.vgg19(img_tensor.unsqueeze(0))
            _, predicted = torch.max(outputs, 1)

            #顯示輸出   
            print(self.classes[predicted])

            #套用sofymax並顯示結果
            s = nn.Softmax(dim=1)
            sig = s(outputs)
            pro_show = sig.detach().numpy()

            #顯示於gui中
            self.predict.setText(f'Predict= {self.classes[predicted]}')
            plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    pro_show[0], tick_label=self.classes)
            plt.ylim((0, 1))
            plt.show()
        except:
            print('imgae type error!')




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_windows = MainWindows()

    main_windows.show()
    sys.exit(app.exec_())
