#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:test.py
@TIME:2019/6/4 14:53
@DES:
'''

if __name__ =="__main__":
    d ={'1':232,'3':324,'2':252}
    sorted(d.items(),key=lambda items:items[1],reverse=True)
    print(d)
    print(d['1'])

    list = [(1, 6), (4, 1), (3, 1), (9, 4)]
    # list.sort(key=lambda x: x[1])
    list.sort(key=lambda x: (x[1], x[0]),reverse=True) #双重排序，先对第二的元素排序，在对第一个元素排序
    print(list)
