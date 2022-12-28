import subprocess
import os
import sys
from time import time
from datetime import datetime, timedelta
from tqdm import trange, tqdm
import schedule

def update_prediction():
    # load model list
    model_path = '/home/ubuntu/2022_VAIV_Dataset/flask/static/models'
    models = os.listdir(model_path)

    kospi = ['vgg16_4', 'efficient_4']
    kosdaq = ['vgg16_kosdaq', 'efficient_kosdaq']
    today = datetime.today().strftime("%Y-%m-%d")

    for kospi_model in kospi:
        print(kospi_model)
        subprocess.call(f'python /home/ubuntu/2022_VAIV_Dataset/try/make_prediction_csv.py -i /home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kospi -s {today} -e {today} -d 224 -o /home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSPI/{kospi_model}.csv -m {model_path}/{kospi_model}.h5 -t Kospi', shell=True)

    for kosdaq_model in kosdaq:
        print(kosdaq_model)
        subprocess.call(f'python /home/ubuntu/2022_VAIV_Dataset/try/make_prediction_csv.py -i /home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kosdaq -s {today} -e {today} -d 224 -o /home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSDAQ/{kosdaq_model}.csv -m {model_path}/{kosdaq_model}.h5 -t Kosdaq', shell=True)


if __name__ == '__main__':
    # 실행 주기 설정
    # Updating Stock Data
    schedule.every().day.at("21:00").do(update_prediction)   # 매일 오후 3시 30분에 update_stock 함수 실행
    #update_prediction()

    # 실행 시작
    while True:
        schedule.run_pending()