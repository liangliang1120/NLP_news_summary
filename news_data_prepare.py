# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 00:16:31 2019
#用于解析新闻文件数据，
1.文字内容提取
2.构建dict(标题：-，内容：-，全文：-)
3.dict.items分词处理
4.dict.items转换数据类型为Sentence对象，为传入SIF计算句子向量做准备
结果为：dict{标题:分词,内容:（分句）,全文
@author: us
"""

