# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:43:07 2019
#用于保存word2vec结果和词频结果
@author: us
"""

import pickle
from gensim.models import word2vec

def my_word2vec(cut_filename):
    mysetence = word2vec.Text8Corpus(cut_filename)
    # model = word2vec.Word2Vec(mysetence, size=300, min_count=1, window=5, hs=5)
    model = word2vec.Word2Vec(mysetence, size=100, min_count=1, window=5, hs=5)
    model.save('./model/zh_wiki_global.model')
    return model

model = my_word2vec('wiki_001.txt')
model_v = {}
for word1 in model.wv.index2word:
    model_v[word1] = model[word1]

def get_word_frequency_all(file_path= 'wiki_cut/wiki_001.txt'):
    # 统计词频
    f= open(file_path,'r',encoding="utf-8")
    word_all = f.read().split()
    fre_dict = {}
    for word_text in word_all:
        if word_text not in fre_dict.keys():
            freq = word_all.count(word_text) 
            fre_dict[word_text] = freq
            print(word_text, freq)
        else:
            pass
    f.close()  
    return fre_dict

#把词频存入文件，需要的时候直接读取，不临时计算词频
fre_dict = get_word_frequency_all()
fileHandle = open ( 'freq_dict_File.file', 'wb' )  
pickle.dump ( fre_dict, fileHandle ) 
fileHandle.close() 

 
#把word2vec存入文件，需要的时候直接读取，不临时计算词向量
fileHandle = open ( 'word2vec_File.file', 'wb' )  
pickle.dump ( model_v, fileHandle ) 
fileHandle.close() 


#读取词频文件 为 dict，及时获取词频数据
#freq = get_word_frequency('这样', file_path='freq_dict_File.file')















