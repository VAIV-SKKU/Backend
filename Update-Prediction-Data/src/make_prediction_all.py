import os
import sys
from time import time
from tqdm import trange, tqdm

import FinanceDataReader as fdr
import pandas as pd
import exchange_calendars as xcals
import os
from datetime import datetime, timedelta
import numpy as np
from tqdm import trange, tqdm
from mpfinance import candlestick2_ochl
import matplotlib.pyplot as plt
from PIL import Image
import yfinance as yf
from csv import writer
import schedule
from pandas_datareader import data
import subprocess
import exchange_calendars as xcals

XKRX = xcals.get_calendar("XKRX") #개장일 가져오기
date_list = XKRX.sessions_in_range('20190101', '20221219').tolist()
dateLength = len(date_list)

date_str_list = []

for date in date_list:
    date_str = date.strftime("%Y-%m-%d")
    date_str_list.append(date_str)

# load model list
model_path = '/home/ubuntu/2022_VAIV_Dataset/flask/static/models'
models = os.listdir(model_path)

kospi = ['efficient_4']
kosdaq = ['vgg16_kosdaq', 'efficient_kosdaq']

dates = ['2022-12-13']

for date in date_str_list:
    subprocess.call(f'python /home/ubuntu/2022_VAIV_Dataset/try/make_prediction_csv.py -i /home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kosdaq -s {date} -e {date} -d 224 -o /home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSPI/efficient_kosdaq.csv -m /home/ubuntu/2022_VAIV_SeoHwan/checkpoint_01/Kosdaq2_EfficientNetB7_224_4_01_2_drop35_batch64_best/model-0001_best.h5 -t Kosdaq', shell=True)

"""
for kospi_model in kospi:
    print(kospi_model)
    for date in dates:
        subprocess.call(f'python /home/ubuntu/2022_VAIV_Dataset/try/make_prediction_csv.py -i /home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kospi -s {date} -e {date} -d 224 -o /home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSPI/{kospi_model}.csv -m {model_path}/{kospi_model}.h5 -t Kospi', shell=True)

#"python /home/ubuntu/2022_VAIV_Dataset/try/make_prediction_csv.py -i /home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kosdaq -s 2022-11-25 -e 2022-11-25 -d 224 -o /home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSDAQ/vgg16_kosdaq.csv -m /home/ubuntu/2022_VAIV_Dataset/flask/static/models/vgg16_kosdaq.h5 -t Kosdaq"


for kosdaq_model in kosdaq:
    print(kosdaq_model)
    for date in dates:
        subprocess.call(f'python /home/ubuntu/2022_VAIV_Dataset/try/make_prediction_csv.py -i /home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kosdaq -s {date} -e {date} -d 224 -o /home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSDAQ/{kosdaq_model}.csv -m {model_path}/{kosdaq_model}.h5 -t Kosdaq', shell=True)
"""