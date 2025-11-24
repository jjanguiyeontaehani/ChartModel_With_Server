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
    - [ ] domain: 비즈니스 로직 앱
        - [ ] services.py: 비즈니스 로직 서비스
            - [X] add_technical_indicators(): 기술적 지표 추가 함수
            - [X] predict_price(): 실시간 종가 예측 함수
    - [ ] services: 공통 서비스 앱
        - [ ] controllers.py: 컨트롤러 로직
            - [X] train_model(): 모델 학습 함수
            - [ ] schedule_model_update(): 모델 업데이트 스케줄링 함수
            - [ ] handle_user_request(): 사용자 요청 처리 함수
            - [ ] send_response(): 응답 전송 함수
        - [ ] tasks.py: 주기적 작업 정의
            - [X] update_models(): 매 시간 모델 업데이트 작업
            - [X] train_on_request(): 사용자 요청 시 모델 학습 작업
    - [ ] view: 메인 페이지 앱
        - [ ] templates: HTML 템플릿
            - [ ] index.html: 메인 페이지 템플릿
        - [X] views.py: 뷰 로직
        - [ ] urls.py: URL 라우팅
    - [ ] repositories: 주가 요청 처리 백엔드 앱
        - [ ] models.py: 데이터베이스
            - [X] modelStatus: 모델 상태 저장
            - [X] stock: 주가 모델 목록
            - [X] stockData: 주가 데이터 저장
            - [ ] celery: Celery 작업 관리
- [ ] requirements.txt: 프로젝트 의존성 목록
- [ ] README.md: 프로젝트 설명 문서
- main.py: 프로젝트 진입점


# 기본 서비스
1. 사용자가 요청한 주식 종목에 대해 모델 학습 후 UI에 실시간 예측 결과 제공
2. 매 1시간마다 종가 예측 모델 업데이트

# 비즈니스 로직
### 종목 학습 및 예측 플로우
1. 사용자가 웹 애플리케이션을 통해 주식 종목을 입력
2. 입력된 종목에 대해 데이터 수집 및 전처리 수행
3. Decoder-only Transformer 모델을 사용하여 종가 예측 모델 학습
4. 학습된 모델정보 및 주식 데이터를 데이터베이스에 저장
5. 초기 학습 완료 시 및 1시간마다 모델을 사용하여 실시간 종가 예측 수행


# 설치 및 실행 방법
아직 프로젝트를 전부 완성하지는 못했지만, 아래의 명령어로 필요한 패키지및 구현된 기능들을 설치하고 실행할 수 있습니다.

### 패키지 설치
```bash
uv uv sync
```

### 웹 애플리케이션 실행
```bash
uv uv sync

cd web_app
python manage.py runserver
```

### 모델 학습
```bash
cd web_app
python manage.py shell

from services.tasks import train_on_request

train_on_request('AAPL') # 'AAPL'을 원하는 주식 심볼로 변경 가능
```
