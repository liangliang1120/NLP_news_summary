# NLP_news_summary
news summary; gensimi;  word2vec;  wiki中文语料; 新闻摘要; SIF; KNN平滑

py文件说明：
    
    gensim_get_cor.py用于获取wiki语料,用了24小时多，不知道是不是我的方法有问题；
    
    jieba_cut.py分词；
    
    jieba_cut_1.py文件拆开后用于分词；
    
    merge_cut_corpus.py合并分词结果；
    
    word2vec.py进行初步词向量训练和部分语料的训练结果；
    
    tsne_word.py用于词向量可视化；
    
    data_prepare_save.py调用gensim训练词向量并统计词频，结果存储于本地文件中；
    
    news_summary.py采用SIF方法训练句子向量，通过KNN平滑提升句子流利度。

    

#

数据说明：
    wiki文件夹是原始语料分成的100个文件；
    wiki_cut文件夹是分成100个文件分词结果；
    wiki_cut_merge.txt是分词结果合并；
    wiki_cut_merge_r19.txt分词结果前19个文件的合并，
        先用这个文件做的词向量训练，因为100个合并太大了；
    wiki_cut_merge_tsne.txt分词结果选取的少量语料，
        先用这个文件做的词向量可视化；
    tsne1.jpg、tsne2.jpg是词向量可视化结果；
    
    word2vec_File.file是训练好的词向量；
    freq_dict_File.file是词向量数据中计算好的词频。
    
    
