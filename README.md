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
└── Update-Prediction-Data      // 매일 주가 데이터, 모델 예측 결과 업데이트
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
- make_image.py 는 매일 주가 차트 이미지를 생성함.
- make_prediction.py 는 모델을 불러와 종목의 상승/하락을 예측.
- predict_csv 폴더에 날짜마다의 종목 상승 예측값이 있음.


## API 설명
### 1) /isvalid [POST]
Login API
로그인 성공 시 My asset 페이지에 회원이 매수한 목록들의 정보가 나타남

</br>

### 2) /updateasset1 [POST]
매수 버튼 클릭 API
매수한 주식의 정보를 DB에 저장

####저장 형식

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

</br>

### 5) /discover [POST, GET]
Today's discover 예측 API

</br>

### 6) /current [POST]
현재가 불러오기

</br>

### 7) /backtest [POST]
모델, KOSPI or KOSDAQ 선택에 따라 과거 상승이라 예측했던 종목들의 csv파일을 불러와 Backtesting 실행

</br>
</br>

## Update Stock Data and Prediction Results Daily
> 매일 장 종료 후 사이트에 사용되는 주가 데이터, 모델 예측 결과 업데이트</br>
> 주가 데이터 업데이트, 이미지 생성, 모델 예측 결과 업데이트 코드는 개별적으로 실행됨
</br>
### Updating Data : Pipeline
```
오후 3시 30분 장 종료 -> 주가 데이터 업데이트 -> 주가 차트 이미지 생성 -> 모델 예측 결과 업데이트
```
</br>

### How to run
+ 주가 데이터 업데이트
    + KOSPI 968종목
        + [Backend/Update-Prediction-Data/src/load_new_data_kospi.py](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/src/load_new_data_kospi.py) 사용
    ```shell script
    $ nohup python load_new_data_kospi.py &
    ```
    + KOSDAQ 1,629종목
        + [Backend/Update-Prediction-Data/src/load_new_data_kosdaq.py](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/src/load_new_data_kosdaq.py) 사용
    ```shell script
    $ nohup python load_new_data_kospi.py &
    ```
    
+ 주가 차트 이미지 생성
    + image size : 224x224
    + channel : 3 (RGB)
    + image features : Open, High, Low, Close (OHLC)
    + KOSPI 968종목
        + [Backend/Update-Prediction-Data/src/load_new_data_kospi.py](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/src/make_image_kospi.py) 사용
    ```shell script
    $ nohup python make_image_kospi.py &
    ```
    + KOSDAQ 1,629종목
        + [Backend/Update-Prediction-Data/src/make_image_kosdaq.py](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/src/make_image_kosdaq.py) 사용
    ```shell script
    $ nohup python make_image_kosdaq.py &
    ```
+ 생성된 이미지에 대한 모델 예측 결과 업데이트
    + KOSPI 968종목, KOSDAQ 1,629종목
        + [Backend/Update-Prediction-Data/src/make_prediction_daily.py](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/src/make_prediction_daily.py), [Backend/Update-Prediction-Data/src/make_prediction_csv.py](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/src/make_prediction_csv.py) 사용
        + [Backend/Update-Prediction-Data/predict_csv/KOSPI](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/predict_csv/KOSPI), [Backend/Update-Prediction-Data/predict_csv/KOSDAQ](https://github.com/VAIV-SKKU/Backend/blob/main/Update-Prediction-Data/predict_csv/KOSDAQ) 폴더의 csv 파일 업데이트
    ```shell script
    $ nohup python make_prediction_daily.py &
    ```
