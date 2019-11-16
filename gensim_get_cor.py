# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:09:51 2019

@author: us
"""

from gensim.corpora.wikicorpus import extract_pages,filter_wiki
import bz2file
import re
#import opencc
from opencc import OpenCC
from tqdm import tqdm
import codecs
cc = OpenCC('t2s')  # convert from Simplified Chinese to Traditional Chinese
wiki = extract_pages(bz2file.open('zhwiki-20190720-pages-articles.xml.bz2'))

def wiki_replace(d):
    s = d[1]
    s = re.sub(':*{\|[\s\S]*?\|}', '', s)
    s = re.sub('<gallery>[\s\S]*?</gallery>', '', s)
    s = re.sub('(.){{([^{}\n]*?\|[^{}\n]*?)}}', '\\1[[\\2]]', s)
    s = filter_wiki(s)
    s = re.sub('\* *\n|\'{2,}', '', s)
    s = re.sub('\n+', '\n', s)
    s = re.sub('\n[:;]|\n +', '\n', s)
    s = re.sub('\n==', '\n\n==', s)
    s = u'【' + d[0] + u'】\n' + s
    return cc.convert(s).strip()

i = 0
f = codecs.open('wiki.txt', 'w', encoding='utf-8')
w = tqdm(wiki, desc=u'已获取0篇文章')
for d in w:
    if not re.findall('^[a-zA-Z]+:', d[0]) and d[0] and not re.findall(u'^#', d[1]):
        s = wiki_replace(d)
        f.write(s+'\n\n\n')
        i += 1
        if i % 100 == 0:
            w.set_description(u'已获取%s篇文章'%i)

f.close()