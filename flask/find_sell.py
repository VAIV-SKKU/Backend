from pathlib import Path
# from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
import sys
# import numpy as np
import shutil
import time
import multiprocessing as mp
# from itertools import product
import exchange_calendars as xcals
import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
p = Path.absolute(Path.cwd().parent.parent)
sys.path.append(str(p))

from Data.stock import Stock, StockMarket
from Data.candlestick import GetName, YoloChart
sys.path.append(str(p / 'YOLO-Train'))
from detect import detect_light, select_device, attempt_load

device = '0'
device = select_device(device)
weights = str(Path.cwd().parent / 'YOLO-Train/runs/train/yolov7ALL-2006-2017train/weights/best.pt')
model = attempt_load(weights, map_location=device)

def default_config():
    config_dict = {
        'Size': [1800, 650],
        'period': 245,
        'candlewidth': 0.8,
        'style': 'default',
        'Volume': False,
        'SMA': [],
        'EMA': [],
        'MACD': [0, 0, 0],
    }

    return config_dict


def default_opt():
    global weights
    opt = {
        'weights': weights,
        'conf_thres': 0.6,
        'device': '0',
        'model': model,
        'imgsz': 640,
        'iou_thres': 0.45,
        'trace': False,
    }
    return opt


def copy_image(tickers, last_date, chart: YoloChart):
    global p
    source = Path.cwd() / 'static' / 'today'
    source.mkdir(parents=True, exist_ok=True)

    notFound = {}
    for ticker in (tickers):
        img = chart.load_chart_path(ticker, last_date)
        try:
            shutil.copy(str(img), str(source / img.name))
        except FileNotFoundError:
            print(img)
            notFound.update({ticker:['FileNotFoundError', 0, '', '']})
        except:
            print('Another Error: ', img)
            notFound.update({ticker:['FileNotFoundError', 0, '', '']})
    return notFound


def detect_list(tickers, last_date, market='Kospi'):
    global weights
    config = default_config()
    name = GetName(root= p / 'Data', **config)
    chart = YoloChart(market=market, name=name, exist_ok=True, root= p / 'Data', **config)
    opt = default_opt()
    opt['weights'] = weights
    sell_tickers = {}
    source = Path.cwd() / 'static' / 'today'
    save_dir = Path.cwd() / 'static' / 'predict'
    source.mkdir(parents=True, exist_ok=True)

    notFound = {}
    files = []
    for ticker in (tickers):
        img = chart.load_chart_path(ticker, last_date)

        if img.exists():
            files.append(str(img))
        else:
            notFound.update({ticker: ['FileNotFoundError', 0, '', '']})
        # try:
        #     shutil.copy(img, str(source / vaiv.path.name))
        # except FileNotFoundError:
        #     print(img)
        #     notFound.update({ticker:['FileNotFoundError', 0, '', '']})
        # except:
        #     print('Another Error: ', img)
        #     notFound.update({ticker:['FileNotFoundError', 0, '', '']})
        #     continue

    # if len(tickers) == len(notFound):
    #     return notFound
    # df = detect_light(**opt, source=source, save_dir=save_dir)

    if not files:
        return notFound

    df = detect_light(**opt, files=files)
    # df = df[df.Signal == 'sell']
    tickers = df.Ticker.tolist()
    probs = df.Probability.tolist()
    signals = df.Signal.tolist()
    starts = df.Start.tolist()
    ends = df.End.tolist()
    ret = {t: [s, p, start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
    ret.update(notFound)
    try:
        shutil.rmtree(str(source))
    except FileNotFoundError:
        pass
    return ret


def detect_all():
    df = pd.read_csv(Path.cwd() / 'static' / 'Stock.csv', index_col=0)
    kospiTickers = df[df['Market'] == 'STK'].index.tolist()
    kosdaqTickers = df[df['Market'] == 'KSQ'].index.tolist()

    today = datetime.today()
    yesterday = today - timedelta(1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    XKRX = xcals.get_calendar("XKRX")
    last_date = XKRX.next_session(yesterday).strftime('%Y-%m-%d')

    # p = Process(detect_list, args=(kospiTickers, last_date, 'Kospi', ))
    Detection = dict()
    start = time.time()
    kospiDict = detect_first(kospiTickers, last_date, 'Kospi')
    # print('Kospi_Dict: ', kospiDict)
    kospiT = time.time()
    kosdaqDict = detect_first(kosdaqTickers, last_date, 'Kosdaq')
    kosdaqT = time.time()

    Detection.update(kospiDict)
    Detection.update(kosdaqDict)

    stock_list = list()
    for ticker, value in Detection.items():
        stock = pd.DataFrame({
            'Ticker': [ticker],
            'FullCode': [df.FullCode.loc[ticker]],
            'Symbol': [df.Symbol.loc[ticker]],
            'Signal': [value[0]],
            'Probability': [value[1]],
            'Start': [value[2]],
            'End': [value[3]],
        })
        stock.set_index('Ticker', drop=True, inplace=True)
        stock_list.append(stock)
    end = time.time()
    print('Kospi Time: ', kospiT - start)
    print('Kosdaq Time: ', kosdaqT - kospiT)
    print('After Detect Time: ', end - kosdaqT)
    detect = pd.concat(stock_list)
    detect.to_csv('./static/Detection.csv')


def detect_MarketFiles(last_date, market):
    global weights
    config = default_config()
    name = GetName(root= p / 'Data', **config)
    chart = YoloChart(market=market, name=name, exist_ok=True, root= p / 'Data', **config)
    opt = default_opt()
    filesPath = chart.path / 'images'
    # fileDetectStart = time.time()
    files = list(map(str, filesPath.glob(f'*{last_date}.png')))
    # print('FileTime: ', time.time() - fileDetectStart)
    opt['weights'] = weights
    df = detect_light(**opt, files=files)
    tickers = df.Ticker.tolist()
    probs = df.Probability.tolist()
    signals = df.Signal.tolist()
    starts = df.Start.tolist()
    ends = df.End.tolist()
    ret = {t: [s, p, start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
    return ret


def detectAllFiles():
    df = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/flask/static/Stock.csv', index_col=0)

    today = datetime.today()
    yesterday = today - timedelta(1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    XKRX = xcals.get_calendar("XKRX")
    last_date = XKRX.next_session(yesterday).strftime('%Y-%m-%d')

    Detection = dict()
    start = time.time()
    kospiDict = detect_MarketFiles(last_date, 'Kospi')
    # print('Kospi_Dict: ', kospiDict)
    kospiT = time.time()
    kosdaqDict = detect_MarketFiles(last_date, 'Kosdaq')
    kosdaqT = time.time()
    Detection.update(kospiDict)
    Detection.update(kosdaqDict)

    stock_list = list()
    for ticker, value in Detection.items():
        stock = pd.DataFrame({
            'Ticker': [ticker],
            'FullCode': [df.FullCode.loc[ticker]],
            'Symbol': [df.Symbol.loc[ticker]],
            'Signal': [value[0]],
            'Probability': [value[1]],
            'Start': [value[2]],
            'End': [value[3]],
        })
        stock.set_index('Ticker', drop=True, inplace=True)
        stock_list.append(stock)
    end = time.time()
    print('Kospi Time: ', kospiT - start)
    print('Kosdaq Time: ', kosdaqT - kospiT)
    print('After Detect Time: ', end - kosdaqT)
    detect = pd.concat(stock_list)
    detect.to_csv('./static/Detection.csv')


def make_process(chart: YoloChart, ticker, last_date, result_dict):
    global p
    stock = Stock(ticker, root=p).load_data()
    stock = stock[stock.index <= last_date]

    condition1 = len(stock) > vaiv.kwargs.get('candle')
    condition2 = last_date in stock.index
    if condition1 & condition2:
        start = stock.index[-245]
        pred = pd.Series({'Start': start, 'End': last_date, 'Date': last_date})
        result_dict['price'][ticker] = stock.loc[last_date, 'Close']
        chart.make_chart(ticker, last_date, pixel=True)
        result_dict['files'].append(str(vaiv.common.image.get('images') / f'{ticker}_{last_date}.png'))
    else:
        result_dict['notFound'].update({ticker: ['FileNotFoundError', 0, 0, '', '']})


# def detect_Test(tickers, last_date, market):
#     s1 = time.time()
#     source = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/RealTime/')
#     source.mkdir(parents=True, exist_ok=True)
#     save_dir = Path('/home/ubuntu/2022_VAIV_Dataset/flask/static/predict/')
#     vaiv = default_vaiv()
#     vaiv.set_kwargs(market=market)
#     vaiv.set_stock()
#     vaiv.set_prediction()
#     vaiv.set_image(source)
#     vaiv.make_dir(common=True, image=True)
#     vaiv.set_labeling()
#     opt = default_opt()
#     opt['weights'] = '/home/ubuntu/2022_VAIV_JSPARK/YOLOv7/yolov7/runs/train/yolov7-Kospi50P_5_3/weights/best.pt'

#     manager = mp.Manager()
#     result_dict = manager.dict()
#     result_dict['notFound'] = manager.dict()
#     result_dict['price'] = manager.dict()
#     result_dict['files'] = manager.list()
#     e1 = time.time()
#     readyT = e1-s1

#     jobs = []
#     s2 = time.time()
#     for ticker in tickers:
#         p = mp.Process(target=make_process, args=(vaiv, ticker, last_date, result_dict))
#         jobs.append(p)
#         p.start()

#     for proc in jobs:
#         proc.join()
#     e2 = time.time()
#     makeT = e2 - s2

#     s4 = time.time()
#     if len(tickers) == len(result_dict['notFound']):
#         return result_dict['notFound']
#     df = detect_light(**opt, source=vaiv.common.image.get('images'), save_dir=save_dir, files=result_dict['files'])
#     e4 = time.time()

#     detectT = e4-s4

#     s5 = time.time()
#     tickers = df.Ticker.tolist()
#     probs = df.Probability.tolist()
#     signals = df.Signal.tolist()
#     starts = df.Start.tolist()
#     ends = df.End.tolist()
#     ret = {t: [s, p, int(result_dict['price'][t]), start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
#     ret.update(result_dict['notFound'])

#     e5 = time.time()
#     returnT = e5 - s5

#     print('Times: ', [readyT, makeT, detectT, returnT])
#     return ret


def detect_first(tickers, last_date, market):
    global weights, p
    config = default_config()
    chart = YoloChart(market=market, name='RealTime', exist_ok=True, root=(Path.cwd() / 'static'), **config)
    save_dir = Path.cwd() / 'static' / 'predict'
    opt = default_opt()
    opt['weights'] = weights

    notFound = {}
    # e1 = time.time()
    # print(f'Ready: {round(e1-s1, 4)}s')

    stock_t = 0
    candle_t = 0
    ticker_count = len(tickers)
    price = {}
    files = []

    jobs = []
    for ticker in tickers:
        # s2 = time.time()
        stock = Stock(ticker, market=market, root= p / 'Data')
        stock.update_data()
        p = mp.Process(target=chart.make_chart, args=(ticker, last_date, ))
        jobs.append(p)
        p.start()
        if chart.load_chart_path(ticker, last_date).exists():
            files.append(chart.load_chart_path(ticker, last_date))
        else:
            notFound.update({ticker: ['FileNotFoundError', 0, 0, '', '']})
            ticker_count -= 1
        # e2 = time.time()
        # stock_t += e2 - s2

    for proc in jobs:
        proc.join()
    # print(f'{ticker_count}/{len(tickers)} Total Stock: {round(stock_t, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Total Candlestick: {round(candle_t, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Avg Stock: {round(stock_t / ticker_count, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Avg Candlestick: {round(candle_t / ticker_count, 2)}s')

    s4 = time.time()
    if len(tickers) == len(notFound):
        return notFound
        # return notFound, [0, 0, 0, 0, 0]
    df = detect_light(**opt, save_dir=save_dir, files=files)
    # e4 = time.time()
    # print(f'Tickers Detect: {round(e4-s4, 2)}s')
    # print(f'{ticker_count}/{len(tickers)} Tickers Detect: {round(e4-s4, 2)}s')
    # detectT = e4-s4

    # s5 = time.time()
    tickers = df.Ticker.tolist()
    probs = df.Probability.tolist()
    signals = df.Signal.tolist()
    starts = df.Start.tolist()
    ends = df.End.tolist()
    ret = {t: [s, p, int(price[t]), start, end] for t, s, p, start, end in zip(tickers, signals, probs, starts, ends)}
    ret.update(notFound)

    # try:
    #     shutil.rmtree(str(source))
    # except FileNotFoundError:
    #     pass

    # e5 = time.time()
    # print(f'{len(tickers)} Tickers Return: {round(e5 - s5, 4)}s')
    # returnT = e5 - s5
    # print('Times: ', [readyT, stock_t, candle_t, detectT, returnT])
    return ret
    # return ret, [readyT, stock_t, candle_t, detectT, returnT]


if __name__ == '__main__':
    tickers = StockMarket(market='Kospi').tickers

    # s0 = time.time()
    # tickers = ['005930', '000020', '095570', '006840', '039570']
    # tickers = ['006390']
    ret = detect_first(tickers=tickers, last_date='2022-12-13', market='Kospi')
    # e0 = time.time()
    # print(ret)
    # print(f'{len(tickers)} Tickers Total: {round(e0-s0, 2)}s')

    # totalT = 0
    # AvgTimes = [0, 0, 0, 0, 0]
    # notFound = 0
    # for ticker in tickers:
    #     s0 = time.time()
    #     ret, times = detect_first(tickers=[ticker], last_date='2022-12-13', market='Kospi')
    #     e0 = time.time()
    #     if ret[ticker][0] == 'FileNotFoundError':
    #         notFound += 1
    #         continue
    #     AvgTimes = [AvgTimes[i] + times[i] for i in range(5)]
    #     totalT += e0 - s0
    # AvgTimes = [t / (len(tickers) - notFound) for t in AvgTimes]
    # totalT /= (len(tickers) - notFound)
    # print('--------------------------------------------------------------------------')
    # print(f'Ready: {round(AvgTimes[0], 4)}s')
    # print(f'{len(tickers) - notFound} Avg Stock: {round(AvgTimes[1], 2)}s')
    # print(f'{len(tickers) - notFound} Avg Candlestick: {round(AvgTimes[2], 2)}s')
    # print(f'{len(tickers) - notFound} Tickers Detect: {round(AvgTimes[3], 2)}s')
    # print(f'{len(tickers) - notFound} Tickers Return: {round(AvgTimes[4], 4)}s')
    # print(f'{len(tickers) - notFound} Tickers Total: {round(totalT, 2)}s')

    # sell_tickers = detect_list(tickers)
    # print(len(sell_tickers))
    # print(sell_tickers)
