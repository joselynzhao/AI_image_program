#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main.py
@TIME:2019/6/3 14:44
@DES:  第一个版本：效果关于模式的识别，只有一个检测错误
'''

import  tensorflow as tf
import  numpy as np
import  math
import matplotlib.pyplot as plt
import matplotlib.image as gImage
from PIL import Image
import cv2
import numpy
from matplotlib import pyplot as plt
import math

data_path="../DATA/"
x_axid = 210
x_bias = 50
window_x = range(x_axid-x_bias, x_axid+x_bias+1)
window_y = range(150, 351)
ss_standerd = 300  #判断是否是腿的方差线
line_width = 10
line_split_model = 500

def init_image(image): #任何图像在进行处理之前首先要进行初始化调整，即该为416*624
    image = image.resize((416, 624))
    image = np.array(image)
    return image


def get_model(image3,image4): #model一共有2个类型，返回0表示衣服不手到裤子里面。返回1表示衣服放在裤子里面
    sp1,sp_line1 = get_split_line(image3)
    sp2,sp_line2 = get_split_line(image4)
    print(sp1,sp2)
    print(sp_line1,sp_line2)
    sp_line1 +=150
    sp_line2 +=150
    if sp_line1+sp_line2 < line_split_model:
        return 0
    else:
        return 1


def index_recover(index):
    new_index = (index-1)*line_width+1
    return new_index

def get_split_line(image): #针对某一张图片 查找腰线 。 供get_model调用..传入的是经过了init之后的图片。
    val_image = image[150:350,x_axid-x_bias:x_axid+x_bias]  #裁剪之后，编码全部从0开始
    plt.imshow(val_image)
    plt.show()
    width,height,path = val_image.shape
    # print(width,height,path)
    squre = []  #记录的one——line 的差距
    for i in range(width):
        sq = 0
        for j in range(height-line_width):
            r1,g1,b1 = val_image[i,j]
            r2,g2,b2 = val_image[i,j+line_width]
            sq += abs(r1-r2) +abs(g1-g2)+abs(b1-b2)
        squre.append(sq)
    #进一步扩宽选择线
    w_squre =[]
    len_squre = len(squre)
    for i in range(0,len_squre-line_width,line_width):
        w_q = 0
        for j in range(line_width):
            w_q+=squre[i+j]
        w_squre.append(w_q)

    max_squre = 0
    max_squre_index = 0
    sec_squre = 0
    sec_squre_index = 0
    for i in range(len(w_squre)):
        if w_squre[i]>max_squre:
            sec_squre = max_squre
            sec_squre_index = max_squre_index
            max_squre = w_squre[i]
            max_squre_index = i
        elif w_squre[i]>sec_squre:
            sec_squre = w_squre[i]
            sec_squre_index = i


    print(max_squre_index,sec_squre_index)
    sec_index = index_recover(sec_squre_index)
    max_index = index_recover(max_squre_index)
    print("new_index:",max_index,sec_index)
    # 检测 squre_index 大于120时的异常情况
    if ((max_squre_index-1)*line_width+1>=120):
        for i in window_x:
            r,g,b = val_image[i,max_squre_index+10]
            mean = (r+b+g)/3.0
            ss = ((r-mean)**2+(g-mean)**2+(b-mean)**2)/3.0
            if(ss<ss_standerd):  # 此时需第二大值。
                return sec_squre,sec_index
            else:
                return max_squre,max_index
    else:
        return max_squre,max_index




def get_input34(i): #i表示要第几个样本
    path3 = data_path + str(i ) + "/input3.JPG"
    print(path3)
    image3 = Image.open(path3)
    image3 = init_image(image3)
    path4 = data_path + str(i ) + "/input4.JPG"
    print(path4)
    image4 = Image.open(path4)
    image4 = init_image(image4)
    return image3,image4

def get_all_input34_and_drew():
    for i in range(4):
        image3,image4 = get_input34(i+1)

        plt.figure()
        plt.subplot(1,2,0)
        plt.title("input3")
        plt.imshow(image3)
        plt.subplot(1, 2, 1)
        plt.title("input4")
        plt.imshow(image4)
        plt.show()



if __name__ =="__main__":
    # get_all_input34_and_drew()
    image3,image4 = get_input34(4)
    model = get_model(image3,image4)
    if model:
        print("衣服放在裤子外面")
    else:
        print("衣服扎在裤子里面")


    # for i in range(4):
    #     image3,image4 = get_input34(i+1)
        # get_split_line(image3)
        # get_split_line(image4)

    # print(window_x)

    # get_all_input34_and_drew()

    # image = Image.open(data_path+"1"+"/input1.JPG")
    # image = image.resize((416,624))
    # image = np.array(image)
    # for i in range(3):
    #     image_one = image[:,:,i]
    #     image = image_one.resize((1000,500))


    # print(height,width,path)
    # for w in range(width):
    #     for h in range(height/2):
    #         r,g,b=image[h,w]
    #         if (r>=50 and r<=255 and g>=0 and g<=50 and b>=0 and b<=50):
    #             image[h,w]=[255,255,255]


    #分别获取三个通道
    # for i in range(3):
    #     image_one = image[:,:,i]
    #     plt.imshow(image_one)
    #     plt.show()

    # plt.figure()



    '''换一个方法读取数据'''


