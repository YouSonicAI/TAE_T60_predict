# -*- coding: utf-8 -*-
"""
@file      :  check_nan.py
@Time      :  2022/8/5 16:55
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

import os
import torch

def Read_Filename(root_dir):
    room_dir_list = os.listdir(root_dir)
    return [root_dir+i for i in room_dir_list if 'pt' in i]


def check_pt(names):
    for path in names:
        print('-------- ', path, ' --------\n')
        data = torch.load(path)
        count = 0
        for _, values in data.items():
            count += 1
            print(count, end=', ')
            image = values[0]['image']
            t60 = values[0]['t60']
            if torch.any(torch.isnan(image)):
                print('-------- ', path, ' --------','-- \nimage nan')
            if torch.any(torch.isnan(t60)):
                print('--------', path, ' --------', 't60 nan')
                print(t60)
    print('count', count)

if __name__ == "__main__":
    path = r"C:/Users/17579/Desktop/Debug/Dataset/Room/"
    names = Read_Filename(path)
    check_pt(names)
