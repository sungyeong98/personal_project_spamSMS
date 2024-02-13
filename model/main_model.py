#학습된 모델을 이용한 실제 모델 코드
#==================================================
#라이브러리 호출
import os
import pandas as pd
import numpy as np
import re
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from googletrans import Translator
#==================================================
#모델 불러오기
script_dir=os.path.dirname(__file__)
relative_path="simple_rnn_model.h5"
file_path=os.path.join(script_dir,relative_path)
model=load_model(file_path)
#==================================================
#문자 내용 가공
trans=Translator()
#실제 스팸문자의 내용을 가져옴
text='[Web발신] 60세 정년퇴임 당일 63,000원 ▶ 923,140원 무료상담 받아보세요! ▼안내▼ http://go9.co/Tyc'
trans_text=trans.translate(text,src='ko',dest='en').text

#clean_text = re.sub(r'[^\w\s]', '', trans_text)
df=pd.DataFrame({'input':[trans_text]})
X=df['input']
#==================================================
#토큰화
relative_path1="../data/processed_data/tokenizer.pickle"
file_path1=os.path.join(script_dir,relative_path1)
with open(file_path1, 'rb') as handle:
    tokenizer = pickle.load(handle)

X_encoded=tokenizer.texts_to_sequences(X)
X_padded=pad_sequences(X_encoded,maxlen=189)
#==================================================
#테스트
result=model.predict(X_padded)[0][0]*100
print(f'해당 문자가 스팸문자일 확률은 {result}% 입니다.')