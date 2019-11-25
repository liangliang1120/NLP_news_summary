# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:54:19 2019

@author: us
"""
import numpy as np
from sklearn.decomposition import PCA
import os

class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    def len(self) -> int:
        return len(self.word_list)


def get_word_frequency(word_text, file_path):
    # 统计词频
    f= open(file_path,'r',encoding="utf-8")
    word_all = f.read().split()
    if word_text in model_v:
        freq = word_all.count(word_text) 
        f.close()  
        print(freq)
    else:
        return 1.0

# sentence_to_vec方法就是将句子转换成对应向量的核心方法
def sentence_to_vec(model_v, sentence_list, embedding_size: int, a: float=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  
        # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        # 这个就是初步的句子向量的计算方法
#################################################
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word.text, 'wiki_cut/wiki_001.txt'))  
            # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))  
            # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  
        # add to our existing re-calculated set of sentences
#################################################
    # calculate PCA of this sentence set,计算主成分
    pca = PCA()
    # 使用PCA方法进行训练
    pca.fit(np.array(sentence_set))
    # 返回具有最大方差的的成分的第一个,也就是最大主成分,
    # components_也就是特征个数/主成分个数,最大的一个特征值
    u = pca.components_[0]  # the PCA vector
    # 构建投射矩阵
    u = np.multiply(u, np.transpose(u))  # u x uT
    # judge the vector need padding by wheather the number of sentences less than embeddings_size
    # 判断是否需要填充矩阵,按列填充
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            # 列相加
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))
    return sentence_vecs


##########################################################

from gensim.models import word2vec
import jieba

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


sentence = '可爱的我喜欢学习新的事物'
sentence_list = list(jieba.cut(sentence, cut_all=False))
sentence_to_vec(model_v, sentence_list, 100, 1e-3)

sentence_list = model_v
embedding_size = 100
a = 1e-3





