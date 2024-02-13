# personal_project_spamSMS

### 프로젝트 개요

해당 개인 프로젝트는 Kaggle의 스팸 문자 데이터셋을 활용하여 스팸 문자를 감지하는 모델을 개발하였습니다. 이를 위해 데이터셋을 분석하고 전처리한 후, 머신러닝 알고리즘을 적용하여 모델을 학습하고 성능을 평가하였습니다. 또한, Googletrans를 이용하여 한국어로 된 문자를 자동으로 번역하여 모델의 예측 결과를 확인하였습니다.

<br>

### 기술 스택

<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

<br>

### 폴더 구조

data/

- 'raw_data': kaggle에서 다운받은 데이터를 저장한 디렉토리
- 'processed_data': 전처리된 데이터와 학습에 사용한 토큰을 저장한 디렉토리

model/

- 'main_model.py': 학습된 모델을 이용한 문자 분류 코드
- 'simple_rnn_model.py': 전처리된 데이터를 이용한 모델 학습 코드

prepro/

- kaggle에서 다운받은 데이터셋을 전처리하기 위한 코드가 저장된 디렉토리