import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# display all columns
pd.set_option('display.max_columns', None)
# display all rows
# pd.set_option('display.max_rows', None)
# from IPython.display import display, HTML


class ETF():
    def __init__(self, ticker):
        self.ticker = ticker  # str

    def get_price_data(self, start_data, end_date):
        '''
        :param start_data: (str)
        :param end_date: (str)
        :return: (df)
        '''
        data = yf.download(self.ticker, start=start_data, end=end_date)
        self.price_data = data['Adj Close']
        return self.price_data

    def cal_ann_return_std(self):
        ann_return = np.log(
            price_data.iloc[-1, :] / price_data.iloc[0, :]) * np.exp(252 / len(price_data))
        std_dev = price_data.std()
        return ann_return, std_dev

    def cal_return_data(self, freq=None):
        '''freq: Daily for default; others: 'W', 'M'
        '''
        if freq is None:
            return_data = np.log(self.price_data.pct_change().dropna() + 1)
        else:
            return_data = np.log(self.price_data.resample(
                freq).ffill().pct_change().dropna() + 1)
        self.return_data = return_data
        return self.return_data

