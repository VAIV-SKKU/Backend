#Short-term Stock Prediction Simulator - Backend
> A simulator that can predict the possible outcomes of inidividual stocks in KOSPI/KOSDAQ

## Framework
Flask


## Getting Started

### Clone Repository

```shell script
$ https://github.com/VAIV-SKKU/Frontend.git

```

### How to Run

```

python demo.py

```
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
├── tree.txt
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

## 


