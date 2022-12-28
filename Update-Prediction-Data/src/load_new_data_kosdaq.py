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

column = ['date','open','high','low','close','volume','ma_5','ma_20','ma_60','ma_120','ma_240']

def df_format(df, last_index):
    # Column 조정
    df = df.reset_index()
    df = df.reset_index()
    df = df.drop(['Change'], axis=1)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = column[:6]
        
    df['ma_5'] = np.full(len(df), np.nan)
    df['ma_20'] = np.full(len(df), np.nan)
    df['ma_60'] = np.full(len(df), np.nan)
    df['ma_120']  = np.full(len(df), np.nan)
    df['ma_240']  = np.full(len(df), np.nan)

    # 날짜 포맷 변경
    df['date'] = df['date'].apply(date_format)

    # 새로운 데이터 추가
    new_data = df.values.tolist()[-1]
    new_data.insert(0, last_index)

    return new_data

def date_format(date):
    return date.strftime("%Y%m%d")

def add_stock(ticker):
    today = datetime.today().strftime("%Y-%m-%d")
    #print(f"today : {today}")
    stock = None
    is_open = None
    try:
        stock_data = pd.read_csv(f'/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kosdaq_Data/{ticker}.csv', index_col=0)
        last_date_str = str(stock_data.iloc[-1]['date']).split('.')[0]
        #print(f"last date : {last_date_str}")
        last_date = datetime.strptime(last_date_str, "%Y%m%d")
        last_index = stock_data.index[-1]

        curr_date = last_date
        while (curr_date < datetime.today()):
            try:
                curr_date = curr_date + timedelta(1)
                df = fdr.DataReader(ticker[1:], curr_date, curr_date)
                last_index += 1
                new_data = df_format(df, last_index)
                with open(f'/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kosdaq_Data/{ticker}.csv', 'a', newline='') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(new_data)
                    f_object.close()
            except:
                continue

    except Exception as e:
        is_open = False

    return stock, is_open


def correction(df, date_format):
    df.columns = map(lambda x: str(x)[0].upper() + str(x)[1:], df.columns)
    delete_col = set(df.columns) - set(column)
    for i in delete_col:
        del df[i]

    if not df.empty:
        # print(df)
        df.Date = df.Date.map(lambda x: correct_date(x, date_format))
    # df['Date'].map(correct_date)

    df = df.replace(0, np.NaN)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[column]

    return df


def correct_date(date, date_format):
    date = str(date)
    date_time = datetime.strptime(date, date_format)
    date = date_time.strftime("%Y-%m-%d")
    return date



def make_candlestick(ticker, date_list):
    df = pd.read_csv(f'/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kosdaq_Data/{ticker}.csv', index_col=0)
    df.reset_index(inplace=True, drop=True)
    #print(df.iloc[4143])
    for date in date_list:
        plt.style.use('dark_background')
        row = None
        # date에 해당하는 row의 index 찾기
        try:
            last_date = date_format(date)
            row = df.loc[df['date'] == float(last_date)]
            index_list = row.index.tolist()
            i = index_list[0] - 20

            c = df.iloc[i:i + 20, :]

            if (len(c) == 20):
                end = date.strftime("%Y-%m-%d")
                name = f'/home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kosdaq/{ticker}_{end}_20_224x224.png'
                fig = plt.figure(figsize=(224 / 100, 224 / 100))
                ax = fig.add_subplot(1, 1, 1)

                quote = pd.DataFrame()
                quote['Date'] = np.arange(0, 20, 1)
                quote = pd.concat([quote, c[['open', 'close', 'high', 'low']]], axis=1)

                candlestick2_ochl(ax, c.open, c.close, c.high, c.low, width=0.7, colorup='#77d879',
                                                colordown='#db3f3f', alpha=None)

                ax.grid(False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.axis('off')

                plt.tight_layout(pad=0)
                fig.set_constrained_layout_pads(w_pad=0, h_pad=0)

                print(name)
                fig.savefig(name)

                pil_image = Image.open(name)
                rgb_image = pil_image.convert('RGB')
                rgb_image.save(name)

                plt.close(fig)
        except Exception as e:
            print(ticker)
            print(f'Date : {date}')
            print(row)
            print(e)
            continue

def check_image(ticker):
    files_path = '/home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kosdaq'
    files = os.listdir(files_path)
    files = [file for file in files if file[0] == 'A']
    count = 0
    date_list = []
    is_there = True
    f_output = open(f'/home/ubuntu/2022_VAIV_Dataset/try/current_image/{ticker}.txt', 'w')
    for file in files:
        if file.split('_')[0] == ticker:
            date_list.append(file.split('_')[1])
            f_output.write(f"{file}\n")
    f_output.close()

    date_list.sort()
    if len(date_list) == 0:
        is_there = False

    return date_list, is_there

def check_date(last_date):
    new_date_list = []
    is_there_date = False
    #last_date_time = datetime.strptime(last_date, "%Y-%m-%d")
    #start_date = date_format(last_date_time)
    end_date = datetime.today().strftime("%Y-%m-%d")

    KOSDAQ = data.get_data_yahoo("^KQ11", last_date, end_date)
    new_date_list = KOSDAQ.index.tolist()
    if len(new_date_list) > 1:
        new_date_list = new_date_list[1:]
        is_there_date = True

    return new_date_list, is_there_date

def update_stock():
    files_path = '/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kosdaq_Data'
    files = os.listdir(files_path)
    files = [file for file in files if file[0] == 'A']
    count = 0

    for file in tqdm(files):
        ticker = file.split('.')[0]
        #print('\nTicker {}/{}\t{}'.format(count, len(files), ticker))

        
        # Update stock data in csv file
        try:
            stock, is_open = add_stock(ticker)
            # 이미지 새로 생성
            if is_open:
                date_list, is_there = check_image(ticker)
                last_date = None
                if is_there:
                    last_date = date_list[-1]
                else:
                    continue
                new_date_list, is_there_date = check_date(last_date)   # KOSDAQ 장이 열린 날짜 중에 date_list의 마지막날 이후의 날짜가 담긴 리스트 (1개 이상일 경우 true 리턴 / 0개일 경우 false도 리턴)
                if is_there_date:
                    make_candlestick(ticker, new_date_list) # is_there_date 가 true일 경우만 실행

            count += 1
        except:
            print(f'Ticker : {ticker}')
            continue

            

    print(count)

    # Update files in try/predict_csv
    today = datetime.today().strftime("%Y-%m-%d")
    kosdaq = ['vgg16_kosdaq', 'efficient_kosdaq']
    model_path = '/home/ubuntu/2022_VAIV_Dataset/flask/static/models'
    for kosdaq_model in kosdaq:
        print(kosdaq_model)
        subprocess.call(f'python /home/ubuntu/2022_VAIV_Dataset/try/make_prediction_csv.py -i /home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kosdaq -s {today} -e {today} -d 224 -o /home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSDAQ/{kosdaq_model}.csv -m {model_path}/{kosdaq_model}.h5', shell=True)

    # Update files in make_graph
    subprocess.call('/home/ubuntu/2022_VAIV_SeoHwan/make_graph/make_2022_csv_kosdaq.py', shell=True)

if __name__ == '__main__':
    # 실행 주기 설정
    # Updating Stock Data
    schedule.every().day.at("16:30").do(update_stock)   # 매일 오후 3시 30분에 update_stock 함수 실행
    #update_stock()

    # 실행 시작
    while True:
        schedule.run_pending()