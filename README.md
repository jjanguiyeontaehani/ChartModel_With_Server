# 미션 주제

### Decoder-only Transformer 기반 주가 예측 모델과 사용자 요청 주가 실시간 예측 웹 구현 챌린지

# 도전 목표

1. PyTorch를 활용한 Decoder-only Transformer 기반 매 시간의 종가 예측 모델 개발
2. Django 웹 프레임워크를 사용한 사용자 요청 주식 학습 및 실시간 예측 웹 애플리케이션 구축


# 기술 스택
### 모델 개발 및 데이터 처리
- Python
- PyTorch
- Pandas
- Numpy
### 웹 애플리케이션 개발
- Django


# 프로젝트 구조
- [X] model: Decoder-only Transformer 모델
    - [X] config: 각종 설정 파일
    - [X] pretrained: 사전 학습된 모델 가중치
    - [X] transformer.py: Transformer 모델 구현
    - [X] train.py: 모델 학습 스크립트
    - [X] predict.py: 모델 예측 스크립트
- [X] data: 주가 데이터셋 및 전처리 스크립트
    - [X] raw: 원본 주가 데이터
    - [X] processed: 전처리된 주가 데이터
    - [X] preprocess.py: 데이터 전처리 스크립트
    - [X] fetch_data.py: 주가 데이터 수집 스크립트
- [ ] web_app: Django 웹 애플리케이션
