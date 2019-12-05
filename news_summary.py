# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 00:16:31 2019
#用于解析新闻，得出摘要
1.文字内容提取
2.构建(标题：-Sentence，内容：（分句）Sentence，全文：-Sentence)
3.dict.items分词处理
4.dict.items转换数据类型为Sentence对象，为传入SIF计算句子向量做准备

@author: us
"""
import jieba
import numpy as np
import pickle
import re
from sklearn.decomposition import PCA
import warnings
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')

def news_summary(input_news, summary_n):

    print('【原文】：{}'.format(input_news[1]))
    
    class Word:
        def __init__(self, text, vector):
            self.text = text
            self.vector = vector
    
    class Sentence:
        def __init__(self, word_list):
            self.word_list = word_list
    
        def len(self) -> int:
            return len(self.word_list)
        
    def get_frequency_dict(file_path='freq_dict_File.file'):
        fileHandle = open ( file_path ,'rb') 
        freq_dict = pickle.load ( fileHandle ) 
        fileHandle.close() 
        return freq_dict
        
        
    def get_word_frequency(word_text, freq_dict):
        if word_text in freq_dict:
            freq = freq_dict[word_text] 
            #print(freq)
            return freq
        else:
            return 1.0
        
    def get_word2vec(file_path='word2vec.file'):
        print('正在载入词向量...')
        fileHandle = open ( file_path ,'rb') 
        word_v = pickle.load ( fileHandle ) 
        fileHandle.close() 
        return word_v
        
    # sentence_to_vec方法就是将句子转换成对应向量的核心方法
    def sentence_to_vec(model_v, allsent, freq_dict, embedding_size: int, a: float=1e-3):
        
        sentence_set = []
        for sentence in allsent:
            vs = np.zeros(embedding_size)  
            # add all word2vec values into one vector for the sentence
            sentence_length = sentence.len()
            # print(sentence.len())
            # 这个就是初步的句子向量的计算方法
    #################################################
            for word in sentence.word_list:
                # print(word.text)
                a_value = a / (a + get_word_frequency(word.text, freq_dict))  
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
        
    
    
    def get_sentence(model_v, train='分化凸显领军自主 电动智能贯穿汽车变革'): 
        
        allsent = []
        for each in train:
            sent1 = list(jieba.cut(each, cut_all=False))
            # print(sent1)
            s1 = []
            for word in sent1:
                # print(word)
                try:
                    vec = model_v[word]
                except KeyError:
                    vec = np.zeros(100)
                s1.append(Word(word, vec))    
            ss1 = Sentence(s1)
            allsent.append(ss1)
        return allsent
        
    #获取已训练好的词向量结果
    model_v = get_word2vec('word2vec_File.file')

    # 1.构建v_t文章标题为Sentence对象类型##########################################
    print('句子格式处理中...')
    v_t = get_sentence(model_v, [input_news[0]])
    
    #分全文为句子
    pattern = r'\?|"|\~|!|#|。|！|【|】'
    content = re.split(pattern, input_news[1])
    try:
        content.remove('')
    except:
        pass
    
    # 2.构建v_i内容句子为Sentence对象类型##########################################
    i = 0
    content_v = {}
    for sen in content:
        content_v[i] = get_sentence(model_v, [sen])
        i = i+1
    
    # 3.构建v_c全文为Sentence对象类型#############################################
    v_c = get_sentence(model_v, [input_news[1]])    
     
    def SIF(v_t, content_v, v_c):
        # v_t,content_v,v_c 按照SIF模型向量化
        print('正在载入词频...')
        freq_dict = get_frequency_dict(file_path='freq_dict_File.file')
        
        print('正在计算句向量...')
        v_t = sentence_to_vec(model_v, v_t, freq_dict, 100, 1e-3)    
            
        for i in range(len(content_v)):
            content_v[i] = sentence_to_vec(model_v, content_v[i], freq_dict, 100, 1e-3) 
            
        v_c = sentence_to_vec(model_v, v_c, freq_dict, 100, 1e-3)    
        return v_t, content_v, v_c
    v_t, content_v, v_c = SIF(v_t, content_v, v_c)
    
    #构建数学模型，输出content_v中与v_c的相关性（要用到向量相似性计算）
    
    #计算两个句向量的余弦相似性值
    def cosine_similarity(vector1=v_c, vector2=v_t):
        
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return 0
        else:
            return round(dot_product / ((normA**0.5)*(normB**0.5)), 2)
    
    # cosine_similarity(v_c[0],v_t[0])
     

    #把与全文相关性最高的n句话挑出来，再按照原本顺序排列，得出摘要 
    print('正在计算句子相似性...')
    sen_vec = {}
    for i in range(len(content_v)):
        sen_vec[i] = cosine_similarity(v_c[0],content_v[i][0])


    X = list(sen_vec.keys())

    def knn_polish(sen_vec, X):
        #只是借鉴了KNN的思想做相似度的平滑，没有算欧氏距离什么的,每个句子和它前一句、后一句做平滑
        for xx in X:
            print(xx)
            if xx > 0 and xx < max(X):
                sen_vec[xx] = (sen_vec[xx - 1] + sen_vec[xx] + sen_vec[xx + 1]) / 3
            elif xx == max(X):
                sen_vec[xx] = (sen_vec[xx - 1] + sen_vec[xx]) / 2
            elif xx == 0:
                sen_vec[xx] = (sen_vec[xx] + sen_vec[xx + 1]) / 2
        return sen_vec

    sen_vec = knn_polish(sen_vec, X)

    sort = []
    temp = sorted(sen_vec.items(), key=lambda x: x[1], reverse=False) 
    for key in temp: 
        sort.append(key[0])  
    keys_sort_n = sort[:summary_n]
    keys_sort_n = sorted(keys_sort_n, reverse=False) 
    
    
    # summary 为输出摘要
    summary = ''
    for n in keys_sort_n:
        summary = summary + content[n]+'。'
    print('摘要完成！')
    print('【摘要】：{}'.format(summary))

if __name__ == '__main__':

    #输入新闻，input_news[0]代表标题，input_news[1]代表内容全文
    #测试可以先不管标题，传进去'标题'不影响
    #测试1
    input_news = ['标题',
    '''
    英国与欧盟的“脱欧”谈判于19日正式开始。然而此时，英国首相特雷莎·梅正面临着空前的政治压力。不久前的大选失利让梅饱受诟病，而尘埃未落的伦敦城西“格伦费尔塔”大火激起的民怨，又给梅“火上浇油”。18日，多家英媒爆出，由于对梅失去信任，保守党党内正在酝酿一场“政变”。
    英国《每日电讯报》称，议会选举败北后，梅对14日“格伦费尔塔”火灾的无情和迟钝反应令她陷入巨大的政治危险。根据伦敦警方17日公布的数字，至少有58人被推定在火灾中丧生，随着搜寻工作继续进行，这一数字可能还会上升。路透社称，如果数字最终确定，“格伦费尔塔”火灾将成为二战后英国发生的最严重火灾。
    英国舆论沉浸在悲伤气氛中的同时，把矛头指向了首相梅的“冷漠”和应对不当。路透社17日称，火灾发生后，英国女王伊丽莎白二世和她的孙子威廉王子16日赴火灾发生地探望灾民和志愿者，女王17日又在自己91岁官方生日庆典上主持了1分钟的默哀仪式，并针对英国近来发生的数起事故“罕见”地呼吁民众“在哀伤中团结起来”。批评者指出，梅在灾后的表现和女王形成“鲜明对比”，显示出梅未能感受到公众情绪，且行动不坚决。
    英国“天空新闻网”17日称，梅在事发后视察火灾现场但未慰问灾民备受指责。作为补救，梅16日来到火灾发生地附近的一座教堂与当地居民见面。但抗议者在教堂外大喊“懦夫”“你不受欢迎”等口号，梅只好在警卫护送下匆匆离去。报道称，除英国女王外，工党领袖科尔宾也在第一时间去了现场并探望幸存者，他们的做法与梅形成“鲜明对比”。
    17日的英媒报道中充满了对梅的讽刺， 英国《每日镜报》头版以“两位领袖的故事”为题对比了梅和女王的灾后表现，并附上两幅截然不同的照片。一幅显示梅慰问火灾生还者受到警卫严密保护，另一幅则是女王与受灾社区居民亲切交谈的场景。
    “梅承认做得不够好”，BBC17日称，为平息怒火，梅当天抽出2小时，在唐宁街会见灾民和志愿者，并主持了一场政府应对火灾的会议。她承诺将亲自监督进行相关公共调查，拿出500万英镑支持灾民，并表示无家可归者将在3周内得到重新安置。
    然而，就在17日下午，首相官邸所在的唐宁街爆发了大规模抗议活动。英国《独立报》称，17日下午，大约1000名抗议者出现在唐宁街，高呼“科尔宾上台”“对抗保守党政府”。这场集会原本是抗议梅率领的保守党与爱尔兰民主统一党谈判联合组阁的，但后来加入了许多对梅应对火灾不力不满的民众。
    路透社称，梅决定提前大选，又未能让保守党在大选中获得绝对多数，已经让英国陷入自一年前“脱欧”公投以来最深刻的政治危机中。专栏作家、前保守党议员帕里斯认为，现在，梅应对火灾的行动表明，她缺乏判断力，“若无法重建公众信任，这个首相当不久”。
    《星期日泰晤士报》18日称，在梅领导的保守党内，人们对梅的信心不断下降，现在一些人甚至已经向她发出最后通牒，要求她在10天内证明她拥有自己所说的“领导能力”，否则就会采取行动赶她下台。报道透露，至少12名保守党议员已打算致函代表保守党后座议员的组织——1922委员会，建议对梅提出不信任动议。《星期日电讯报》18日也引述一些保守党“脱欧”派资深人士的话说，如果梅在即将展开的“脱欧”谈判中，背离原来的“硬脱欧”计划，他们就会立即对梅的领导权提出挑战。“脱欧”派议员警告说，任何让英国留在欧盟内的企图，或任何“偏航”的做法，都将在“一夜之间”触发“政变”。
    '''
    ]
    
    
    #测试2
    input_news = ['标题','整车,预计下半年回暖同时自主分化趋势延续。上半年乘用车整体表现偏淡,前5月同比增长3.2%,增速放缓的主要原因在于购置税优惠政策调整导致部分需求预支以及部分消费人群热情削弱。预计下半年乘用车销量回升,全年增速有望实现5%-7%,主要在于1)中长期,居民购买力持续攀升将保持市场消费力和意愿维持旺盛状态;2)短期,7.5%的购置税优惠政策仍有刺激作用。同时,前5月自主品牌依旧强势,市占率较2016年底再提升1pp,一线自主如吉利、上汽自主、广汽自主前5月累计销量增速分别达89%、119%、61%,远高于整体增速,更强于二三线自主,由于一线自主竞争力强,且乘用车市场短期不会恢复火爆局面,我们认为这一阶段自主品牌分化将延续,竞争格局将更明朗。\
    零部件,自主共振和国产替代的成长路径将更清晰。自主共振型的零部件厂商,因为各类原因发展历程相对缓慢,但由于配套近年崛起迅速的领军自主,从而销量、利润,进而研发、管理都将得到快速提升,经过升级迭代将直面三资竞争对手。这个路径上,配套量(绑定快速成长的自主车企)和单车配套价值量方面(专注于高附加值零部件)能得以提升零部件商将更为受益。国产替代型的零部件厂商,一般早年在合资车企或大型三资一级供应商的扶持下供应合资车企,数年以来研发、管理体系都逐渐健全。其中的佼佼者不仅可以对内继续拓展自主客户,对外还可以继续向海外客户延伸。由于同时具备技术实力、成本优势、中国企业独特的灵活性,成功的可能性非常大。这一类公司目前多是细分领域龙头,并体现出强者愈强的趋势。\
    电动化、智能化持续贯穿汽车变革,近期建议关注全球新能源汽车市场同步增长的趋势,同时今年将是全球范围内的爆款车元年,建议把握以特斯拉为首的爆款车产业链投资机会。电动化方面,2017年1-5月新能源汽车累计产量13.0万辆,同比增长12%。其中新能源商用车市场还未完全启动,但随客车的夏季需求小高峰到来、电动物流车商业模式重新确定需求,预计新能源商用车需求将从三季度开始平稳启动。上半年新能源乘用车销量保持增长,非限购城市的需求持续出现。建议把握全球新能源市场同步增长的趋势,重点关注今年新能源乘用车爆款车及其产业链机会,尤其特斯拉产业链。6月起特斯拉Model 3的量产预期,以及明年巨头的加入预期将持续带来催化效应。智能化方面,2017年上半年智能驾驶领域政策继续完善,支持力度延续,但各大巨头开始集结站队,形成不同阵营。目前智能驾驶正在步入订单验证阶段,以摄像头和雷达为主的预警类ADAS 有望优先放量,坚定看好执行端企业布局智能驾驶领域。\
    投资建议:\
    2017年下半年汽车行业处于自主分化加剧与电动智能带来的双变革中。\
    整车方面,在自主车企产品力、品牌力持续提升的同时,伴随着分化加剧,推荐一线自主标的【吉利汽车H+上汽集团+广汽集团H】,建议关注众泰汽车以及经销商标的广汇汽车。\
    零部件方面,自主共振类型的供应商有望实现量价齐升,国产替代类型的供应商将强者愈强,推荐优质零部件标的【拓普集团+新泉股份+精锻科技+新坐标+万里扬】,建议关注宁波高发、星宇股份、奥特佳。\
    新能源汽车方面,建议关注全球新能源汽车市场同步增长的趋势,同时今年将是全球范围内的爆款车元年,建议把握以特斯拉为首的爆款车产业链投资机会。\
    推荐特斯拉产业链标的【拓普集团】,以及新能源整车企业【众泰汽车、江淮汽车】,建议关注特斯拉产业链标的广东鸿图。\
    智能网联方面,目前智能驾驶正在步入订单验证阶段,坚定看好执行层投资机会,建议关注智能化标的拓普集团、网联化标的宁波高发。\
    风险提示:宏观经济走弱、汽车消费情绪转淡、新能源政策执行力度低于预期、智能网联汽车订单落地低于预期。（天风证券 崔琰）']
    #标题：分化凸显领军自主 电动智能贯穿汽车变革
    
    
    #摘要summary_n条句子
    summary_n = 9
    result = news_summary(input_news, summary_n)