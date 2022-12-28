from pykrx import stock
from datetime import datetime, timedelta
import pandas as pd
from tqdm import trange, tqdm

# today = datetime.today().strftime("%m%d")

# ticker_list = stock.get_market_ticker_list('22' + today, market='KOSDAQ')
# print(ticker_list)

df = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSPI/vgg16_4.csv', index_col=0)

for row in tqdm(df.values):
    ticker = None
    date = None
    try:
        ticker = row[1] # ticker
        ticker = ticker[1:]
        date = row[2]   # 날짜

        date_time = datetime.strptime(row[2], "%Y-%m-%d")
        year = date_time.strftime("%Y")
        month = date_time.strftime("%m")
        year_sub = year[2:]
        if year_sub == '22' and month == '11':
            date_str = year_sub + date_time.strftime("%m%d")

            ticker_list = stock.get_market_ticker_list(date_str, market='KOSPI')
            if ticker not in ticker_list:
                print(f"date : {date_str}, ticker : {ticker}")
    except Exception as e:
        continue