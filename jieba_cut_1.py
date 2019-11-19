# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:22:38 2019

@author: us
"""
import jieba

def jieba_cut(filename, cut_filename):
    with open(filename, 'rb') as f:
        mycontent = f.read()
        jieba_content = jieba.cut(mycontent, cut_all=False)
        print('is cutting1...')
        final_file = ' '.join(jieba_content)
        print('is cutting2...')
        final_file = final_file.encode('utf-8')

    with open(cut_filename, 'wb+') as cut_f:
        cut_f.write(final_file)

    
for i in range(2,101):
    n = str(i)
    s = n.zfill(3)
    print("wiki/wiki_{}.txt".format(s) )
    jieba_cut("wiki/wiki_{}.txt".format(s), "wiki_cut/wiki_{}.txt".format(s))


