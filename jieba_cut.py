# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:22:38 2019

@author: us
"""
import jieba
from gensim.models import word2vec

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
        
        

def my_word2vec(cut_filename):
    
    mysetence = word2vec.Text8Corpus(cut_filename)
    
    #model = word2vec.Word2Vec(mysetence, size=300, min_count=1, window=5, hs=5)
    
    model = word2vec.Word2Vec(mysetence, size=100, min_count=1, window=5, hs=5)
    
    model.save('./model/zh_wiki_global.model')
    
    return model
    

jieba_cut('wiki/wiki_001.txt', 'wiki_cut.txt')
#
#
# model = my_word2vec('wiki_cut.txt')
#
#
#
# for key in model.similar_by_word(u'爸爸', topn=10):
#
#         print(key)
#
# print('*****************')
#
# for key in model.similar_by_word(u'对不起', topn=10):
#
#         print(key)