#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main2.py
@TIME:2019/6/3 20:13
@DES:
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
from operator import itemgetter, attrgetter


data_path="../DATA/"
x_axid = 190
x_bias = 30
y_up = 150
y_botton = 330
window_x = range(x_axid-x_bias, x_axid+x_bias+1)
window_y = range(y_up, y_botton+1)
ss_standerd = 300  #判断是否是腿的方差线
line_width = 10
line_split_model = 500
ratio = 3  #放松和模糊比例
white_botton = 220

def init_image(image): #任何图像在进行处理之前首先要进行初始化调整，即该为416*624
    image = image.resize((416, 624))
    image = np.array(image)
    return image


def get_model(image3,image4): #model一共有2个类型，返回0表示衣服不放到裤子里面。返回1表示衣服放在裤子里面
    line11,line12 = get_split_line(image3)
    line21,line22 = get_split_line(image4)
    # print(sp1,sp2)
    print(line11,line12)
    print(line21,line22)

    #进行位置恢复
    line11 += y_up
    line12 += y_up
    line21 += y_up
    line22 += y_up

    if line11+line21 > line_split_model:
        return 0
    else:
        return 1




def index_recover(index):
    new_index = (index)*ratio
    return new_index

def belong_white(r,g,b):
    if r in range(white_botton,256) and g in range(white_botton,256) and b in range(white_botton,256):
        return 1
    else:
        return 0


def get_white_line(image): #传入的应该是s_image
    height,width,path = image.shape
    print("get_white_line", image.shape,height)
    i = height-1
    while(1):
        count = 0
        get = 0
        for k in range(width):
            r,g,b = image[i,k]
            if belong_white(r,g,b):
                count+=1
                if count>3: #找到了一个多个白色像素连续的地方
                    i=i-1
                    get = 1
                    break
            else:
                count=0
        if not get:
            return i  #表示白色出现在i的下一行。

        if i< height/3:
            print("查找白色分割线遇到麻烦了")
            return  i









def get_split_line(image): #针对某一张图片 查找腰线 。 供get_model调用..传入的是经过了init之后的图片。
    val_image = image[y_up:y_botton,x_axid-x_bias:x_axid+x_bias]  #裁剪之后，编码全部从0开始
    plt.imshow(val_image)
    plt.show()
    height,width,path = val_image.shape
    # print(width,height,path)
    temp =Image.fromarray(val_image)
    temp = temp.resize((width/ratio,height/ratio)) #宽、高
    s_image = np.array(temp)
    plt.imshow(s_image)
    plt.show()
    print(s_image.shape)

    #接下里需要对逐个像素进行计算
    height,width, path = s_image.shape
    # print(width,height,path)
    differ=[] # 记录的相邻两行的差距
    for i in range(height-1):
        sq = 0
        for j in range(width):  #遍历所有的点
            r1, g1, b1 = s_image[i, j]
            r2, g2, b2 = s_image[i+1, j]
            sq += get_squre([r1,r2]) + get_squre([g1,g2]) + get_squre([b1,b2])
        # differ[i] = sq #添加字典元素
        differ.append((i,sq))

    # 对differ进行排序
    # sorted(differ.items(), key=lambda x: x[1],reverse=True)
    differ.sort(key=lambda x: (x[1], x[0]), reverse=True)  # 双重排序，先对第二的元素排序，在对第一个元素排序

    white_line = get_white_line(s_image)
    # white_line = index_recover(white_line)
    print("white line:",white_line)
    temp_i = 0
    while(1):
        if(differ[temp_i][0]+5>=white_line):
            temp_i+=1
        elif differ[temp_i][0]<=5:
            temp_i+=1
        else:
            break
        if temp_i>=len(differ):
            temp_i= 0
            break
    #         index_y = differ[temp_i][0]+ 5
    #     else:
    #         index_y = height-2
    #     sum = 0
    #     for key in range(width):
    #         r,g,b=s_image[key,index_y]
    #         sq = get_squre([r,g,b])
    #         sum+=sq
    #     mean_sq = sum/float(width)
    #     if mean_sq<1000: #这个值待调整
    #         # 这条line就要被抛弃
    #         temp_i+=1
    #     else:
    #         break
    print("temp_i:",temp_i)
    # max = differ[temp_i]
    temp_j = temp_i+1
    while(1):
        if abs(differ[temp_j][0]-differ[temp_i][0])<10:
            temp_j+=1
        elif differ[temp_j][0]+20 > white_line:
            temp_j+=1
        elif differ[temp_j][0]<=5:
            temp_j+=1
        else:
            break
        if temp_j>=len(differ):
            temp_j= temp_i+1
            break

    # 看一下回复后的white_line
    print(index_recover(white_line))
    max_differ = differ[temp_i]
    sec_differ = differ[temp_j]
    max_differ_index = index_recover(max_differ[0])
    # max_differ = max_differ[1]
    sec_differ_index = index_recover(sec_differ[0])
    # sec_differ = sec_differ[1]
    print("key :", max_differ,sec_differ)

    print("key max index :", max_differ_index)
    print("key sec index:", sec_differ_index)


    # max_squre = 0
    # max_squre_index = 0
    # sec_squre = 0
    # sec_squre_index = 0
    # for i in range(len(differ)):
    #     if differ[i] > max_squre and abs(i-max_squre_index)>=10:
    #         sec_squre = max_squre
    #         sec_squre_index = max_squre_index
    #         max_squre = differ[i]
    #         max_squre_index = i
    #     elif differ[i]>max_squre and abs(i-max_squre_index)<10:
    #         max_squre = differ[i]
    #         max_squre_index = i
    #     elif differ[i] > sec_squre and abs(i-sec_squre_index)>=10:
    #         sec_squre = differ[i]
    #         sec_squre_index = i

    # print("old_index:",max_squre, sec_squre)
    # print("old_index:",max_squre_index, sec_squre_index)
    # sec_index = index_recover(sec_squre_index)
    # max_index = index_recover(max_squre_index)
    # print("new_index:", max_index, sec_index)
    # # 检测 squre_index 大于120时的异常情况
    # if (max_index >= 120):
    #     for i in window_x:
    #         r, g, b = val_image[i, max_squre_index + 10]
    #         ss=get_squre([r,g,b])
    #         if (ss < ss_standerd):  # 此时需第二大值。
    #             return sec_squre, sec_index
    #         else:
    #             return max_squre, max_index
    # else:
    #     return max_squre, max_index
    return max_differ_index,sec_differ_index

def get_all_windows():
    for i in range(4):
        image3,image4 = get_input34(i+1)
        image3 = image3[y_up:y_botton,x_axid-x_bias:x_axid+x_bias]  #裁剪之后，编码全部从0开始
        image4 = image4[y_up:y_botton,x_axid-x_bias:x_axid+x_bias]  #裁剪之后，编码全部从0开始
        plt.figure()
        plt.subplot(1, 2, 0)
        plt.title("input3")
        plt.imshow(image3)
        plt.subplot(1, 2, 1)
        plt.title("input4")
        plt.imshow(image4)
        plt.show()


def get_squre(a):
    len_a = len(a)
    mean = 0
    for i in range(len_a):
        mean += a[i]
    mean = mean/float(len_a)
    squre = 0
    for i in range(len_a):
        squre +=(a[i]-mean)**2
    squre = squre/float(len_a)
    return squre


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
    # get_all_windows()

    image3,image4 = get_input34(4)
    model = get_model(image3,image4)
    if model:
        print("衣服扎在裤子里面")
    else:
        print("衣服放在裤子外面")
