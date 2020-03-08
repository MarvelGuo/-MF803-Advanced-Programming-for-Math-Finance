import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# display all columns
pd.set_option('display.max_columns', None)

class ETF():
    def __init__(self, ticker):
        self.ticker = ticker  # str

    def get_price_data(self, start_date, end_date):
        '''
        :param start_data: (str)
        :param end_date: (str)
        :return: (df)
        '''
        data = yf.download(self.ticker, start=start_date, end=end_date)
        self.price_data = data['Adj Close']
        return self.price_data

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

    def cal_ann_return_std(self):
        self.ann_return = self.return_data.mean() * 252
        self.std_dev = self.return_data.std() * np.sqrt(252)
        return self.ann_return, self.std_dev

    def cov_matrix(self, data, matrix_type):

        if matrix_type == 'cov':
            return data.cov()
        elif matrix_type == 'corr':
            return data.corr()
        else:
            raise ValueError('Input Must be Correlation or Covariance')


