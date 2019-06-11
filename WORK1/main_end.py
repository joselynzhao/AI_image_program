#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main_end.py
@TIME:2019/6/7 00:16
@DES:
'''

import  numpy as np
from PIL import Image
import matplotlib.pyplot as plt


catch_range_x=[100,350]
size_for_re = [416,624]
seed_local_up = [150,110]
label = 1
model_line = 250
window_x=[50,200]

seed_m=[(250,75),(300,75),(280,75),(350,75)]
pthresh=100
rthresh = 60
seed_up = [150,125]
shift_list = [0,-5,-15,+15]


def get_figure(path):
    image  = Image.open(path)
    image = image.resize((416,624))
    image = np.array(image)
    image = image[:,catch_range_x[0]:catch_range_x[1]]
    plt.imshow(image)
    plt.title("get_figure")
    plt.show()
    return image


def rgb2gray(rgb):
    r, g, b=rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray=0.2989*r + 0.5870*g + 0.1140*b
    return gray

def seed_fill01(img,seed,pthresh,rthresh):
    '''
    seed：种子
    pthresh: 种子像素阈值
    rthresh: 相邻像素阈值
    '''
    img = rgb2gray(img) #RGB-Gray
    h,w = img.shape
    stack_list = []
    labels = np.zeros([h,w]) #先将labels 全部填充为0
    seedy,seedx = seed[0],seed[1]
    stack_list.append((seedy,seedx)) #将种子压入栈内
    #搜索状态空间
    while len(stack_list)!=0:
        cur_i = stack_list[-1][0] #i在h方向上
        cur_j = stack_list[-1][1] #j在w方向上
        labels[cur_i][cur_j]=label
        stack_list.remove(stack_list[-1]) #将栈顶pop出来
        # 判断边界条件，防止数组溢出
        if (cur_i == 0 or cur_i == h - 1 or cur_j == 0 or cur_j == w - 1):
            continue
        if (np.abs(int(img[cur_i-1,cur_j])-int(img[seedy,seedx]))<pthresh) and \
                (np.abs(int(img[cur_i - 1, cur_j]) - int(img[cur_i, cur_j])) < rthresh) and \
                labels[cur_i-1,cur_j]!=label:
            stack_list.append((cur_i-1,cur_j))
        if (np.abs(int(img[cur_i,cur_j-1])-int(img[seedy,seedx]))<pthresh) and \
                (np.abs(int(img[cur_i, cur_j-1]) - int(img[cur_i, cur_j])) < rthresh) and \
                labels[cur_i,cur_j-1]!=label:
            stack_list.append((cur_i, cur_j-1))
        if (np.abs(int(img[cur_i, cur_j + 1]) - int(img[seedy, seedx])) < pthresh) and \
                (np.abs(int(img[cur_i, cur_j+1]) - int(img[cur_i, cur_j])) < rthresh) and \
                labels[cur_i, cur_j + 1] != label:
            stack_list.append((cur_i, cur_j + 1))
        if (np.abs(int(img[cur_i+ 1, cur_j ]) - int(img[seedy, seedx])) < pthresh) and \
                (np.abs(int(img[cur_i + 1, cur_j]) - int(img[cur_i, cur_j])) < rthresh) and \
                labels[cur_i+ 1, cur_j ] != label:
            stack_list.append((cur_i+ 1, cur_j ))

    plt.imshow(labels, cmap='gray_r')
    plt.title("labels")
    plt.show()
    return labels

def isAcceptable(img, point, candidate, pixelThreshold, regionThreshold, labels, seeds):
    x,y = point
    a,b = candidate
    i,j = seeds
    if (np.abs(int(img[x,y])-int(img[a,b]))<pixelThreshold) and (np.abs(int(img[i,j])-int(img[a,b]))<regionThreshold) and labels[a,b] != 1:
        return True
    else:
        return False

def pix_seed_fill(img, seed, pthresh, rthresh):
    # 获取图像的尺寸
    img = img[:,:,0] #选择红色通道作为处理对象
    h, w = img.shape
    # 定义待扩展栈
    stack_list = []
    head_list = []
    # stack_list.append(seeds)
    head_list.append(seed)
    # 定义已扩展区域
    labels = np.zeros([h, w])
    count = 0
    head_list_len = 5000
    # 搜索状态空间
    while len(stack_list) + len(head_list) != 0:
        if count % head_list_len == 0:
            head_len = len(head_list)
            if head_len > head_list_len:
                stack_list += head_list[0:head_list_len]
                head_list = head_list[head_list_len:head_len]
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
        # 判断边界条件，防止数组溢出
        if (cur_i == 0 or cur_i == h - 1 or cur_j == 0 or cur_j == w - 1):
            continue
        if isAcceptable(img, [cur_i, cur_j], [cur_i - 1, cur_j], pthresh, rthresh, labels, seed):
            head_list.append((cur_i - 1, cur_j))
            labels[cur_i - 1][cur_j] = 1
        if isAcceptable(img, [cur_i, cur_j], [cur_i, cur_j - 1], pthresh, rthresh, labels, seed):
            head_list.append((cur_i, cur_j - 1))
            labels[cur_i][cur_j - 1] = 1
        if isAcceptable(img, [cur_i, cur_j], [cur_i + 1, cur_j], pthresh, rthresh, labels, seed):
            head_list.append((cur_i + 1, cur_j))
            labels[cur_i + 1][cur_j] = 1
        if isAcceptable(img, [cur_i, cur_j], [cur_i, cur_j + 1], pthresh, rthresh, labels, seed):
            head_list.append((cur_i, cur_j + 1))
            labels[cur_i][cur_j + 1] = 1

    # 当待探索栈为空，返回区域标记
    plt.imshow(labels, cmap='gray_r')
    plt.title("labels")
    plt.show()
    return labels

def get_model(image,window_x):
    '''
    :param image: 获得的label图像，只含0和label
    :param window_x: 横向遍历的范围 [50,200]
    :return: model 0表示衣服在裤子外面，1表示衣服在裤子里面
    '''
    h,w = image.shape
    line = 0
    bool_find = 0
    for i in range(h): #从上到下进行遍历
        for j in range(window_x[0],window_x[1],1):
            if image[i,j]== label:
                line = i
                bool_find = 1
                break

        if bool_find==1:
            break

    print("line:",line)
    if line >= model_line:
        return 0
    else:
        return 1

def combine_image(image1,image2,model,k):
    '''
    :param image1: input1
    :param image2: input2
    :param image2_label:  input2的裤子抠图
    :param model:  0表示外扎腰，1表示内扎腰
    :return:  合成后的图像
    '''
    h, w, p = image1.shape
    if model == 0:# 在外扎腰的情况下  取衣服贴到裤子上， 即1放到2中
        labels = pix_seed_fill(image1, seed_up, pthresh, rthresh)  # 从image1中取出上衣
        for i in range(350):
            for j in range(w):
                if labels[i][j]==label: #需要抠出来的部分
                    if j+shift_list[k]>=w:
                        pass
                    else:
                        image2[i][j+shift_list[k]][:]=image1[i][j][:]
        return image2
    else: #model = 1的情况
        labels = pix_seed_fill(image2, seed_m[k], pthresh, rthresh)  # 从image2中取出裤子
        for i in range(150,h,1):
            for j in range(w):
                if labels[i][j] == label:
                    if j+shift_list[k]>=w:
                        pass
                    else:
                        image1[i,j+shift_list[k],:] = image2[i,j,:]
        return image1



if __name__ =="__main__":
    for i in range(0,4,1):
        path_get_model = "00%d-input4.JPG"%(i+1)

        # 获取图片
        # 对图片进行初始化，截图，降低分辨率
        image = get_figure(path_get_model)
        labels = pix_seed_fill(image,seed_m[i],pthresh,rthresh)

        #接下里确定组合模式
        model = get_model(labels,window_x)
        print(i,model)

        path_get_input1 = "00%d-input1.JPG"%(i+1)
        path_get_input2 = "00%d-input2.JPG"%(i+1)

        # 获取图片
        # 对图片进行初始化，截图，降低分辨率
        image1 = get_figure(path_get_input1)
        image2 = get_figure(path_get_input2)

        # 组合衣服和裤子
        image = combine_image(image1,image2,model,i)

        #保存并显示图像
        save_path = "./00%d-output.png"%(i+1)
        plt.imshow(image)
        plt.title("00%d-output"%(i+1))
        plt.savefig(save_path)
        plt.show()

