from gensim.models import word2vec

def my_word2vec(cut_filename):
    mysetence = word2vec.Text8Corpus(cut_filename)
    # model = word2vec.Word2Vec(mysetence, size=300, min_count=1, window=5, hs=5)
    model = word2vec.Word2Vec(mysetence, size=100, min_count=1, window=5, hs=5)
    model.save('./model/zh_wiki_global.model')
    return model

model = my_word2vec('wiki_cut_merge_r19.txt')


for key in model.similar_by_word(u'爸爸', topn=10):
        print(key)
print('*****************')
for key in model.similar_by_word(u'对不起', topn=10):
        print(key)
print('*****************')
for key in model.similar_by_word(u'勇敢', topn=10):
    print(key)

'''
使用1/5语料训练结果
('妈妈', 0.8562487363815308)
('爷爷', 0.7699095010757446)
('太太', 0.7695075273513794)
('女朋友', 0.7594177722930908)
('奶奶', 0.7559012770652771)
('老婆', 0.7490440011024475)
('女孩', 0.7444847822189331)
('朋友', 0.7439483404159546)
('女人', 0.7431105971336365)
('小孩', 0.7310429811477661)
*****************
('我', 0.7551761269569397)
('谢谢', 0.7436092495918274)
('啊', 0.7319035530090332)
('再见', 0.7313019037246704)
('你', 0.7176611423492432)
('妳', 0.713086724281311)
('不好意思', 0.7085527181625366)
('干嘛', 0.7025700807571411)
('亲爱', 0.6914868354797363)
('爱过', 0.6837902069091797)
*****************
('坚强', 0.7980549335479736)
('冷静', 0.747090220451355)
('善良', 0.7392765879631042)
('勇气', 0.7337237596511841)
('温柔', 0.7333598136901855)
('热情', 0.7332494854927063)
('自信', 0.7246155738830566)
('冷酷', 0.7208274602890015)
('真诚', 0.7165604829788208)
('骄傲', 0.7147254943847656)
'''