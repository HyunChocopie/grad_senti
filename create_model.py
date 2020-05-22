# coding: utf-8

import sys
sys.path.append('..')


import pandas as pd

#datas=pd.read_json('train_docs.json')
#datas=pd.read_json('sent_docs.json')
datas=pd.read_excel('d3.xlsx')

import numpy as np

from gensim.models import Word2Vec

xlist=[]
for data in datas[0]:
    data=data.replace("'","")
    morlist=data.split(", ")
    xlist.append(morlist)
    

wvmodel=Word2Vec(xlist,size=200,window=3,min_count=5,sg=1)
wvmodel.save("511wvmodel.model")

print("*")

sen_vecs=[]
for data in xlist:
    
    tmp_vec=np.zeros((1,200))
    for d in data:
        
        if d in wvmodel.wv.vocab:
            
            tmp_vec=tmp_vec+wvmodel.wv[d]
    tmp_vec=tmp_vec/len(tmp_vec)
    sen_vecs.append(tmp_vec)


from sklearn.model_selection import train_test_split
#X=np.array(sen_vecs)
#Y=np.array(labels)

X=sen_vecs
X=np.array(X)
X=X.reshape((60000,200))

Y=datas[1]
#Y=list(map(lambda y: 0 if y=='negative' else 1,datas[1]))

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,shuffle=True)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import optimizers


model=Sequential()

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))



sgd=optimizers.SGD(lr=0.001)
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train, batch_size=100, epochs=1000, validation_data=(x_test, y_test))
#loss, accuracy, precision, recall = model.evaluate(x_test, y_test)
model.save('511model.h5')

#print("accuracy : ",accuracy)
#print("precision : ",precision)
#print("recall : ",recall)

from performance_measure import print_precision_recall

y_pred=model.predict_classes(x_test)
y_pred=to_categorical(y_pred)
print_precision_recall(y_test,y_pred)

'''

converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()
tflite_model_name='mymodel_kkma.tflite'
open(tflite_model_name,'wb').write(tflite_model)
'''