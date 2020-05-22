#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:13:45 2020

@author: hyunjeong
"""

import pandas as pd

data1=pd.read_excel('sentiment_dataset.xlsx')
#data2=pd.read_excel('t)

from konlpy.tag import Okt
from konlpy.tag import Kkma
okt=Okt()
kkma=Kkma()

import json
import os

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]
    #return ['/'.join(t) for t in kkma.pos(doc)]

sent_docs=[]
for r,p in zip(data.review,data.polarity):
    if p=='negative':
        label="0"
    elif p=='positive':
        label="1"
    sent_docs.append((tokenize(r), label))
    
    
#with open('senti_docs.json', 'w', encoding="utf-8") as make_file:
#    json.dump(sent_docs, make_file, ensure_ascii=False, indent="\t")
    
    
with open('senti_docs.json', 'w', encoding="utf-8") as make_file:
    json.dump(sent_docs, make_file, ensure_ascii=False, indent="\t")