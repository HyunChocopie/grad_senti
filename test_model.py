#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:14:31 2020

@author: hyunjeong
"""

import pandas as pd
#lex=pd.read_excel('dataset55.xlsx')


from konlpy.tag import Kkma,Okt
kkma = Kkma()
okt = Okt()

testsen="떡볶이가 맛있는ㅇ데 너무 멀어서 시켜먹기 좀 그래"

#tempmors=kkma.pos(testsen)
#tempmors=okt.pos(testsen)


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]
    #return ['/'.join(t) for t in kkma.pos(doc)]

testmors=tokenize(testsen)

import tensorflow as tf

from gensim.models import Word2Vec
wvmodel = Word2Vec.load('511wvmodel.model')

import numpy as np
testvec=np.zeros((1,200))
for m in testmors:
    if m in wvmodel.wv.vocab:
        testvec=testvec+wvmodel.wv[m]




model = tf.keras.models.load_model('511model.h5')

yhat=model.predict(testvec)

print(yhat)

negprob=yhat[0][0]
if negprob>0.6:
    print("부정")
elif negprob<0.6 and negprob>0.4:
    print("중립")
else:
    print("긍정")


