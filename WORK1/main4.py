#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main4.py
@TIME:2019/6/5 20:34
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
import cv2

from main3 import *

data_path="../DATA/"
catch_range_x=[100,350]
size_for_re = [416,624]
seed_local_up = [150,110]



def seed_fill(img):
    ret,img = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
    label = 100
    stack_list = []
    h,w = img.shape
    for i in range(1,h-1,1):
        for j in range(1,w-1,1):
            if (img[i][j] == 255):
                img[i][j] = label
                stack_list.append((i,j))
                while len(stack_list)!=0:
                    cur_i = stack_list[-1][0]
                    cur_j = stack_list[-1][1]
                    img[cur_i][cur_j] = label
                    stack_list.remove(stack_list[-1])
                    #######四邻域，可改为八邻域
                    if (img[cur_i-1][cur_j] == 255):
                        stack_list.append((cur_i-1,cur_j))
                    if (img[cur_i][cur_j-1] == 255):
                        stack_list.append((cur_i,cur_j-1))
                    if (img[cur_i+1][cur_j] == 255):
                        stack_list.append((cur_i+1,cur_j))
                    if (img[cur_i][cur_j+1] == 255):
                        stack_list.append((cur_i,cur_j+1))
    cv2.imwrite('./result.jpg',img)
    cv2.imshow('img',img)
    cv2.waitKey()


def function():
    image = Image.open(data_path+"1/input3.JPG")
    image.resize((416,624))
    image = np.array(image)
    print(image.shape)
    image = image[:,catch_range_x[0]:catch_range_x[1]]
    print(image.shape)

def rgb2gray(rgb):
    r, g, b=rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray=0.2989*r + 0.5870*g + 0.1140*b
    return gray

def get_figure(path):
    image  = Image.open(path)
    image = image.resize((416,624))
    # '''转换为灰度图'''
    # image = image.convert("L")
    image = np.array(image)
    # plt.imshow(image)
    # plt.show()

    image = image[:,catch_range_x[0]:catch_range_x[1]]
    return image

def my_seed_fill(img,seed_local):
    # h,w,p = img.shape
    # for pp in range(p):
    #     plt.imshow(img[:,:,pp])
    #     plt.title(pp)
    #     plt.show()


    #如果我只考虑红色通道
    # img = img[:,:,1]
    # plt.imshow(img)
    # plt.show()

    '''获取种子点的信息，以便设置阈值'''
    img = rgb2gray(img)  #将其设置为 灰度
    color = img[seed_local[0],seed_local[1]]
    print(color)
    ret,img = cv2.threshold(img,color+10,255,cv2.THRESH_BINARY_INV)
    label = 100
    stack_list =[]
    h,w = img.shape
    # print("h,w:",h,w)
    for i in range(1,h-1,1):
        for j in range(1,w-1,1):
            if(img[i][j]==255):
                img[i][j]=label
                stack_list.append((i,j))
                while len(stack_list)!=0:
                    cur_i = stack_list[-1][0]
                    cur_j = stack_list[-1][1]
                    img[cur_i][cur_j]=label
                    stack_list.remove(stack_list[-1])
                    if cur_i ==0 or cur_i == h-1 or cur_j==0 or cur_j==w-1:
                        continue
                    if (img[cur_i-1][cur_j] == 255):
                        stack_list.append((cur_i-1,cur_j))
                    if (img[cur_i][cur_j-1] == 255):
                        stack_list.append((cur_i,cur_j-1))
                    if (img[cur_i+1][cur_j] == 255):
                        stack_list.append((cur_i+1,cur_j))
                    if (img[cur_i][cur_j+1] == 255):
                        stack_list.append((cur_i,cur_j+1))

    plt.imshow(img,cmap='gray_r')
    plt.show()
    return img

def isAcceptable(img, point, candidate, pixelThreshold, regionThreshold, labels, seeds):
    x,y = point
    a,b = candidate
    i,j = seeds
    if (np.abs(int(img[x,y])-int(img[a,b]))<pixelThreshold) and (np.abs(int(img[i,j])-int(img[a,b]))<regionThreshold) and labels[a,b] != 1:
        return True
    else:
        return False

def he_seed_fill(img, seeds, pthresh, rthresh):
    # 获取图像的尺寸
    h, w = img.shape
    # 定义待扩展栈
    stack_list = []
    head_list = []
    # stack_list.append(seeds)
    head_list.append(seeds)
    # 定义已扩展区域
    labels = np.zeros([h, w])

    count = 0

    head_list_len = 5000

    # 搜索状态空间
    while len(stack_list) + len(head_list) != 0:
        if count % head_list_len == 0:
            print("expand %d, stack %6d head %6d" % (count, len(stack_list), len(head_list)))

            head_len = len(head_list)
            # stack_len = len(stack_list)
            if head_len > head_list_len:
                stack_list += head_list[0:head_list_len]
                head_list = head_list[head_list_len:head_len]
            print("stack %6d head %6d" % (len(stack_list), len(head_list)))

        if len(head_list) == 0 and len(stack_list) >= head_list_len:
            head_list = stack_list[len(stack_list) - head_list_len:len(stack_list)]
            stack_list = stack_list[0:len(stack_list) - head_list_len]
        elif len(head_list) == 0 and len(stack_list) < head_list_len:
            head_list = stack_list
            stack_list = []

        # 获取需要扩展的点的坐标
        cur_i, cur_j = head_list[-1]
        # 标记当前点为已探索
        labels[cur_i][cur_j] = 1
        # 从栈中弹出该点
        head_list.remove(head_list[-1])
        count += 1
        ### 四邻域扩展，可改为八邻域 ###

        # 判断边界条件，防止数组溢出
        if (cur_i == 0 or cur_i == h - 1 or cur_j == 0 or cur_j == w - 1):
            continue

        if isAcceptable(img, [cur_i, cur_j], [cur_i - 1, cur_j], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i - 1, cur_j))
            labels[cur_i - 1][cur_j] = 1
        if isAcceptable(img, [cur_i, cur_j], [cur_i, cur_j - 1], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i, cur_j - 1))
            labels[cur_i][cur_j - 1] = 1
        if isAcceptable(img, [cur_i, cur_j], [cur_i + 1, cur_j], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i + 1, cur_j))
            labels[cur_i + 1][cur_j] = 1
        if isAcceptable(img, [cur_i, cur_j], [cur_i, cur_j + 1], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i, cur_j + 1))
            labels[cur_i][cur_j + 1] = 1
        # if isAcceptable(img, [cur_i,cur_j], [cur_i-1,cur_j-1], pthresh, rthresh, labels, seeds):
        #     stack_list.append((cur_i-1,cur_j-1))
        #     labels[cur_i-1][cur_j-1] = 1
        # if isAcceptable(img, [cur_i,cur_j], [cur_i-1,cur_j+1], pthresh, rthresh, labels, seeds):
        #     stack_list.append((cur_i-1,cur_j+1))
        #     labels[cur_i-1][cur_j+1] = 1
        # if isAcceptable(img, [cur_i,cur_j], [cur_i+1,cur_j-1], pthresh, rthresh, labels, seeds):
        #     stack_list.append((cur_i+1,cur_j-1))
        #     labels[cur_i+1][cur_j-1] = 1
        # if isAcceptable(img, [cur_i,cur_j], [cur_i+1,cur_j+1], pthresh, rthresh, labels, seeds):
        #     stack_list.append((cur_i+1,cur_j+1))
        #     labels[cur_i+1][cur_j+1] = 1
    # 当待探索栈为空，返回区域标记
    return labels



if __name__ =="__main__":
    # image3,image4 = get_input34(1)
    # print(image3.shape)
    # print(image4.shape)
    path = "../DATA/1/input4.JPG"

    # 获取图片
    # 对图片进行初始化，截图，降低分辨率
    image = get_figure(path)
    image = my_seed_fill(image,seed_local_up)


    #




