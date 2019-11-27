# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:54:19 2019
#切分句子，只按照句号
#后需要把词频记录下来直接查，word2vec也要记录，不能每次调用再去算，存到数据库再取会不会快一点


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
    if word_text in word_all:
        freq = word_all.count(word_text) 
        f.close()  
        #print(freq)
        return freq
    else:
        return 1.0

# sentence_to_vec方法就是将句子转换成对应向量的核心方法
def sentence_to_vec(model_v, allsent, embedding_size: int, a: float=1e-3):
    sentence_set = []
    for sentence in allsent:
        vs = np.zeros(embedding_size)  
        # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        print(sentence.len())
        # 这个就是初步的句子向量的计算方法
#################################################
        for word in sentence.word_list:
            print(word.text)
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


# sentence = '可爱的小美喜欢学习新的事物'
# sentence_list = list(jieba.cut(sentence, cut_all=False))
train = ['世界首批智能机器警犬惊现美国马萨诸塞州街头执勤，吓坏民权组织'
    ,'据美国媒体11月25日报道，马萨诸塞州警方从今年8月开始，不动声色地使用世界上首批智能机器警犬在街头执勤，这些机器警犬装备了人工智能程序，可以探查可疑包裹，追踪犯罪嫌疑人藏身地点，还可以轻易地解锁开门。'
    ,'WBUR首先报道了马萨诸塞州警方使用机器警犬执勤，称警方将其作为“移动遥控监测装置”使用。'
    ,'警方的录像资料表明，机器警犬“斑点”采用人工智能程序和计算机识别系统的机械臂可以轻而易举地解锁开门。'
    ,'据悉，“斑点”装备了一只机械臂和一个弱光环境摄像头，可以自动行走，也可以遥控操作。'
    ,'这款机器警犬在开发阶段就屡屡爆出惊人成就，例如采用人工智能程序和计算机识别系统的机械臂可以轻而易举地解锁开门。'
    ,'波士顿动力公司表示，“斑点”专用于非暴力的公共安全执勤。']


allsent = []
for each in train:
    sent1 = list(jieba.cut(each, cut_all=False))
    print(sent1)
    s1 = []
    for word in sent1:
        print(word)
        try:
            vec = model[word]
        except KeyError:
            vec = np.zeros(100)
        s1.append(Word(word, vec))

    ss1 = Sentence(s1)
    allsent.append(ss1)

sentence_vectors = sentence_to_vec(model_v, allsent, 100, 1e-3)
# sentence_to_vec(model_v, sentence_list, 100, 1e-3)
sentence_list = model_v
embedding_size = 100
a = 1e-3



