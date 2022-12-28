import sys
import os
from collections import defaultdict
import numpy as np
import scipy.misc
import imageio
import cv2
import pickle
import argparse
import time
from copyreg import constructor
# import tensorflow as tf
from datetime import timedelta
from datetime import datetime
import exchange_calendars as xcals
from pykrx import stock
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
from mplfinance.original_flavor import candlestick2_ochl, volume_overlay
from temp import find_sell_date

# ,index,Ticker,Date,Predicted,Probability
# 0,301,A000020,2019-01-02,0,0.5907316
# 1,66,A000040,2019-01-02,0,0.7023688

"""
def calculate_profit(day, t, market):
  year = day.split('-')[0]
  csv_date = day.replace('-', '')
  profit = None
  profit_calcul = 'True'
  try:
    stock_data = pd.read_csv(f'/home/ubuntu/2022_VAIV_SeoHwan/make_graph/{year}_{market}_data/{t}.csv',encoding='CP949')
    now_row = stock_data[stock_data['Date']==int(csv_date)]
    index = int(now_row.values[0][0])
    pred_row = stock_data[stock_data['Unnamed: 0']==index+5]
    now_close = float(now_row.values[0][3])
    pred_close = float(pred_row.values[0][3])
    profit = round((pred_close*0.9975-now_close)/now_close *100, 3)
    #print(profit)
  except Exception as e:
    #print(e)
    profit = 0
    profit_calcul = 'False'
  #print(profit_calcul)
  return profit, profit_calcul
"""

def calculate_profit(day, t, market):
  year = day.split('-')[0]
  csv_date = day.replace('-', '')
  profit = None
  profit_calcul = 'True'
  try:
    # 현재가 불러오기
    currentClose = None
    try:
        s = requests.Session()
        url = "https://finance.naver.com/item/main.nhn?code=" + t[1:]
        resp = s.get(url)
        currentClose = resp.text.split("<dd>현재가 ")[1].split(' ')[0].replace(",","")
    except:
        print("current close price was not loaded.")
        profit = 0
        profit_calcul = 'False'
        return profit, profit_calcul
    
    stock_data = pd.read_csv(f'/home/ubuntu/2022_VAIV_SeoHwan/make_graph/{year}_{market}_data/{t}.csv',encoding='CP949')
    now_row = stock_data[stock_data['Date']==int(csv_date)]
    index = int(now_row.values[0][0])
    now_close = float(now_row.values[0][3])
    pred_close = float(currentClose)
    profit = round((pred_close*0.9975-now_close)/now_close *100, 3)
    #print(profit)
  except Exception as e:
    #print(e)
    profit = 0
    profit_calcul = 'False'
  #print(profit_calcul)
  return profit, profit_calcul

def discover(stockMarket, date, numOfStock):
  stocklist = []

  # Index 0 : ticker ex) A00020
  # Index 1 : 종목명 ex) 삼성전자
  # Index 2 : 선택한 날짜의 종가
  # Index 3 : [[‘VGG16’, 상승예측 여부, 상승확률], [‘EfficientNet’, 상승예측 여부, 상승확률]]
  # - 상승예측 여부 : 1 (상승 예측) / 0 (상승 예측 x)
  # - 상승확률 : 70.2 (백분율, 소수 둘째자리에서 반올림, % 붙이지 않음)
  # Index 4 : 선택한 날짜의 종가와 오늘 날짜의 종가를 기반으로 계산한 수익률 ex) 1.0% (백분율, 소수 둘째자리에서 반올림, % 붙임)
  # - 오늘 날짜의 추천 종목을 보는 경우 None을 리턴, 그 외의 경우 수익률을 리턴

  # row1 = ['A005930', '삼성전자', 90000, [['VGG16', 0, 60.0], ['EfficientNet', 1, 80.2]], None]
  # row2 = ['A000660', 'SK 하이닉스', 54020, [['VGG16', 1, 65.1], ['EfficientNet', 1, 78.6]], None]

  # stocklist.append(row1)
  # stocklist.append(row2)

  probs = [0.625, 0.625, 0.675, 0.55]
  # csv 파일 불러오기
  suffix = None
  offset = None
  stockDir = None
  market = None
  csv_directory = f'/home/ubuntu/2022_VAIV_Dataset/try/predict_csv/{stockMarket}'
  if stockMarket == 'KOSPI':
    suffix = '_4.csv'
    offset = 0
    stockDir = 'Kospi'
    market = 'stcok'
  else:
    suffix = '_kosdaq.csv'
    offset = 2
    stockDir = 'Kosdaq'
    market = 'kosdaq'
  vgg_df = pd.read_csv(f'{csv_directory}/vgg16{suffix}', index_col=0)
  effi_df = pd.read_csv(f'{csv_directory}/efficient{suffix}', index_col=0)
  # 이상한 행 제거
  vgg_df = vgg_df[~vgg_df['Probability'].str.contains("Probability", na=False, case=False)]
  effi_df = effi_df[~effi_df['Probability'].str.contains("Probability", na=False, case=False)]

  # 열별로 타입 통일
  vgg_df = vgg_df.astype({'Probability':'float'})
  effi_df = effi_df.astype({'Probability':'float'})
  vgg_df = vgg_df.astype({'Predicted':'int'})
  effi_df = effi_df.astype({'Predicted':'int'})

  # 날짜 맞는 것만 뽑고 columns, index 정리하기
  vgg_df = vgg_df.loc[vgg_df['Date'] == date]
  effi_df = effi_df.loc[effi_df['Date'] == date]
  vgg_df = vgg_df[['Ticker', 'Date', 'Predicted', 'Probability']]
  effi_df = effi_df[['Ticker', 'Date', 'Predicted', 'Probability']]

  # 중복 제거
  vgg_df.drop_duplicates(['Ticker', 'Date'], keep='first', inplace=True, ignore_index=False)
  effi_df.drop_duplicates(['Ticker', 'Date'], keep='first', inplace=True, ignore_index=False)

  vgg_df.reset_index(inplace=True, drop=True)
  effi_df.reset_index(inplace=True, drop=True)

  # 1. prediction 값이 1인 것만 뽑기
  vgg_one = vgg_df.loc[vgg_df['Predicted'] == 1]
  effi_one = effi_df.loc[effi_df['Predicted'] == 1]

  # 2. prediction 값이 1이면서 probability가 threshold 이상인 종목만 뽑기
  vgg_prob = vgg_one.loc[vgg_one['Probability'] >= probs[0 + offset]]
  effi_prob = effi_one.loc[effi_one['Probability'] >= probs[1 + offset]]
  vgg_prob.reset_index(inplace=True, drop=True)
  effi_prob.reset_index(inplace=True, drop=True)

  #print(vgg_prob)
  #print(effi_prob)

  # 각각 확률값 높은 상위 20종목만 뽑기
  vgg_top20 = None
  effi_top20 = None
  vgg_prob.sort_values(by=['Probability'], axis=0, inplace=True, ascending=False)
  vgg_prob.reset_index(inplace=True, drop=True)
  effi_prob.sort_values(by=['Probability'], axis=0, inplace=True, ascending=False)
  effi_prob.reset_index(inplace=True, drop=True)
  if len(vgg_prob) > numOfStock:
    vgg_top20 = vgg_prob.iloc[0:numOfStock]
  else:
    vgg_top20 = vgg_prob.copy()
  if len(effi_prob) > numOfStock:
    effi_top20 = effi_prob.iloc[0:numOfStock]
  else:
    effi_top20 = effi_prob.copy()

  vgg_top20['vggSignal'] = np.ones(len(vgg_top20))
  effi_top20['effiSignal'] = np.ones(len(effi_top20))
  vgg_top20.rename(columns={'Probability':'vggProbability'}, inplace=True)
  effi_top20.rename(columns={'Probability':'effiProbability'}, inplace=True)

  vgg_top20 = vgg_top20[['Ticker', 'vggProbability', 'vggSignal']]
  effi_top20 = effi_top20[['Ticker', 'effiProbability', 'effiSignal']]
  # outline merge, ticker 기준
  merged_df = pd.merge(vgg_top20, effi_top20, on='Ticker', how='outer')
  merged_df.reset_index(inplace=True)
  merged_df = merged_df.fillna(0)

  merged_df.sort_values(by=['vggSignal', 'effiSignal'], axis=0, inplace=True, ascending=False)
  #print(merged_df)

  t_date = datetime.strptime(date, "%Y-%m-%d")
  s_date = t_date.strftime("%Y%m%d")
  # top20 종목 리스트 (vgg, effi 둘 중 하나는 무조건 _df 에서 값 가져와야 함)
  for row in merged_df.values:
    rowlist = []
    ticker = row[1]
    vggProb = row[2]
    vggSig = row[3]
    effiProb = row[4]
    effiSig = row[5]
    profit = None
    sumProb = None
    #print(f"ticker : {ticker}")
    #stock_df = pd.read_csv(f'/home/ubuntu/2022_VAIV_Dataset/Stock_Data/{stockDir}_Data/{ticker}.csv', index_col=0)
    stock_df = pd.read_csv(f'/home/ubuntu/2022_VAIV_SeoHwan/make_graph/2022_{market}_data/{ticker}.csv',encoding='CP949', index_col=0)

    stockrow = stock_df.loc[stock_df['Date'] == float(s_date)]
    close = stockrow['Close'].item()
    #print(close)
    #print(ticker)
    if effiSig == 0.0:
      thatrow = effi_df.loc[effi_df['Ticker'] == ticker]
      effiProb = thatrow['Probability'].item()
    if vggSig == 0.0:
      thatrow = vgg_df.loc[vgg_df['Ticker'] == ticker]
      vggProb = thatrow['Probability'].item()
    
    rowlist.append(ticker)
    #print(ticker[1:])
    stockname = stock.get_market_ticker_name(ticker[1:])
    #print(stockname)
    rowlist.append(stockname)  # 종목명
    #print(stock.get_market_ticker_name("078935"))
    rowlist.append(close)
    vgglist = ['VGG16', vggSig, round(vggProb*100, 1)]
    effilist = ['EfficientNet', effiSig, round(effiProb*100, 1)]
    rowlist.append([vgglist, effilist])
    
    sumProb = vggProb + effiProb
    if not vggSig == 0.0:
      if not effiSig == 0.0:
        sumProb += 500

    rowlist.append(sumProb)

    stocklist.append(rowlist)

  #print(stocklist)
  # stocklist를 sumProb 기준으로 정렬
  stocklist.sort(key=lambda stocklist: stocklist[4], reverse=True)

  return stocklist

def backtest(stockMarket, date):
  stocklist = []

  

  return stocklist

# result = discover('KOSDAQ', '2022-12-13', 20)
# print(result)