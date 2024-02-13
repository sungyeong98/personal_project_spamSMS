#스팸문자를 분류하기 위한 모델 코드
#==================================================
#라이브러리 호출
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
import pickle
#==================================================
#데이터 불러오기
script_dir=os.path.dirname(__file__)
relative_path="../data/processed_data/spam.csv"
file_path=os.path.join(script_dir,relative_path)
df=pd.read_csv(file_path,encoding='latin-1')
#==================================================
#train,test설정
X=df['v2']
y=df['v1']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
#==================================================
#토큰화
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded=tokenizer.texts_to_sequences(X_train)

word_to_index=tokenizer.word_index

X_train_padded=pad_sequences(X_train_encoded,maxlen=189)
#==================================================
#모델 설정
embedding_dim,hidden_units=32,32
vocab_size=len(word_to_index)+1

model=Sequential()
model.add(Embedding(vocab_size,embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
RNN_model=model.fit(X_train_padded,y_train,epochs=4,batch_size=64,validation_split=0.2)
#==================================================
#모델 테스트
X_test_encoded=tokenizer.texts_to_sequences(X_test)
X_test_padded=pad_sequences(X_test_encoded,maxlen=189)
result=model.evaluate(X_test_padded,y_test)
#==================================================
#모델 저장
relative_path1="simple_rnn_model.h5"
file_path1=os.path.join(script_dir,relative_path1)
model.save(file_path1)
#==================================================
#토큰 저장
relative_path2="../data/processed_data/tokenizer.pickle"
file_path2=os.path.join(script_dir,relative_path2)
with open(file_path2, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)