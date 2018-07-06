# coding:utf-8
__author__ = 'XJX'
__date__ = '2018.07.05'

"""
description:
    利用正则表达式提取关键词
"""

import os
import re
import codecs
import sys
import importlib
import msgpack
import fnmatch


def extract(path):
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    print(filelist)

    ans = []

    for files in filelist:  # 遍历所有文件
        if not fnmatch.fnmatch(files, '*.txt'):
            continue

        try:
            f = open(path + files, 'r+', encoding='utf8')

            for line in f.readlines():
                results = re.findall("<([a-zA-Z]*)>([^<]*)</[a-zA-Z]*>", line)
                for result in results:
                    print(result)
                    # ans.append(result)

            f.close()
            print(path + files + ' done!')

        except Exception:
            print('Error:' + path + files)
            continue




if __name__ == '__main__':
    extract('./data/抽取准确文本（820人）/')
