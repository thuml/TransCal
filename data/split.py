#!/usr/bin/env python
#coding:utf-8

from sklearn.model_selection import train_test_split
def split_data(file_name):
    source_list = open(file_name).readlines()
    source_train, source_val = train_test_split(source_list, test_size=0.2)
    print(len(source_train))
    print(len(source_val))
    source_train_file_name = file_name.replace('list', 'train_list')
    source_val_file_name = file_name.replace('list', 'val_list')

    source_train_file = open(source_train_file_name, "w")
    for line in source_train:
        source_train_file.write(line)

    source_val_file = open(source_val_file_name, "w")
    for line in source_val:
        source_val_file.write(line)

# file_name_list = ['office/amazon_list.txt', 'office/webcam_list.txt', 'office/dslr_list.txt',
# 'office-home/Art_list.txt', 'office-home/Clipart_list.txt', 'office-home/Product_list.txt','office-home/Real_World_list.txt',
# 'image-clef/b_list.txt', 'image-clef/c_list.txt', 'image-clef/i_list.txt','image-clef/p_list.txt',
# ]
file_name_list = ['office-home/Art_list.txt', 'office-home/Clipart_list.txt', 'office-home/Product_list.txt','office-home/Real_World_list.txt']

for file_name in file_name_list:
    split_data(file_name)
