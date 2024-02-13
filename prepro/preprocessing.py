#스팸문자 데이터셋을 전처리하는 코드
#=======================================================
#라이브러리 호출
import os
import pandas as pd
import numpy as np
#=======================================================
#데이터 불러오기
script_dir=os.path.dirname(__file__)
relative_path="../data/raw_data/spam.csv"
file_path=os.path.join(script_dir,relative_path)
df=pd.read_csv(file_path,encoding='latin-1')
#=======================================================
#필요없는 열 삭제
del_col=['Unnamed: 2','Unnamed: 3','Unnamed: 4']
df=df.drop(del_col,axis=1)
#=======================================================
#열 데이터 변경
pd.set_option('future.no_silent_downcasting', True)
df['v1']=df['v1'].replace(['ham','spam'],[0,1])
df.drop_duplicates(subset=['v2'],inplace=True)
#=======================================================
#타겟 단어 저장
'''
target=['free', 'award', 'urgent', 'cash', 'mobile', 'tone', 'txt',  'guaranteed', 'bonus', 'voucher' , '鶯', 'www']
for i in target:
    df[i]=df['text'].apply(lambda x:1 if i.lower() in x.lower() else 0)
'''
#=======================================================
#가공된 데이터 저장
relative_path1="../data/processed_data/spam.csv"
file_path1=os.path.join(script_dir,relative_path1)
df.to_csv(file_path1)