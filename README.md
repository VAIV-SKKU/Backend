# Short-term Stock Prediction Simulator - Backend
> A simulator that can predict the possible outcomes of inidividual stocks in KOSPI/KOSDAQ

<br />
<br />

## Framework
Flask

<br />

## Getting Started

### Clone Repository
```shell script
$ https://github.com/VAIV-SKKU/Frontend.git

```
<br />

### How to Run
```
python demo.py
```
<br />

## 파일 구조

```
.
├── flask
│   ├── autotrading.py
│   ├── backtesting.py
│   ├── buy_daily.py
│   ├── demo.py
│   ├── find_sell.py
│   ├── node_modules
│   │   ├── bootstrap
│   │   ├── jquery
│   │   └── mobiscroll
│   ├── predict_newDemo.py
│   └── stockdata.py
└── Update-Prediction-Data
    ├── predict_csv
    │   ├── KOSDAQ
    │   │   ├── efficient_kosdaq.csv
    │   │   └── vgg16_kosdaq.csv
    │   └── KOSPI
    │       ├── efficient_4.csv
    │       └── vgg16_4.csv
    └── src
        ├── check_image.py
        ├── check_ticker.py
        ├── load_new_data_kosdaq.py
        ├── load_new_data_kospi.py
        ├── make_image_kosdaq.py
        ├── make_image_kospi.py
        ├── make_prediction_all.py
        ├── make_prediction_csv.py
        ├── make_prediction_daily.py
        ├── make_prediction_period.py
        └── utils
            ├── dataset_sampling.py
            ├── dataset_testing.py
            ├── get_data.py
            ├── __init__.py
            ├── png2pickle.py
            └── __pycache__
```
- 서버는 demo.py 로 실행
- 


## API 설명
### 1) /isvalid [POST]
Login API
로그인 성공 시 My asset 페이지에 회원이 매수한 목록들의 정보가 나타남

</br>

### 2) /updateasset1 [POST]
매수 버튼 클릭 API
매수한 주식의 정보를 DB에 저장

저장 형식

```
stock_info = {
        "market" : market,
        "ticker" : ticker,   #fixed
        "name" : name,   #fixed
        "buy date" : buy_date,   #fixed
        "buy count" : int(buy_count),   #fixed
        "buy close" : int(buy_close),   #fixed
        "buy total" : int(buy_total),   #fixed
    }
```

</br>

### 3) /updateasset2 [POST]
My asset 클릭 시 총 자산 정보, 현재가 정보를 포함한 asset list 보여줌

</br>

### 4) /updateasset3 [POST]
매도 API


### 5) /discover [POST, GET]
Today's discover 예측 API

### 6) /current [POST]
현재가 불러오기

### 7) /backtest [POST]
모델, KOSPI or KOSDAQ 선택에 따라 과거 상승이라 예측했던 종목들의 csv파일을 불러와 Backtesting 실행


