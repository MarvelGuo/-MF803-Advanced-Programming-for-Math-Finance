#!/usr/bin/env python
# coding: utf-8

# In[27]:


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


def cov_matrix(return_data, matrix_type):

    if matrix_type == 'cov':
        return return_data.cov()
    elif matrix_type == 'corr':
        return return_data.corr()
    else:
        raise ValueError('Input Must be Correlation or Covariance')


def rolling_coef(return_data, win, column):
    '''
    :param return_data:
    :param win: length of rolling window
    :param column: which column is calculated with
    '''
    rolling_coef = return_data.rolling(win).corr(return_data[column])

    fig = plt.figure(dpi=100)
    rolling_coef.iloc[:, 1:].dropna().plot(figsize=(12, 8), ax=plt.gca())
    plt.ylabel('Coefficient', fontsize = 20 )
    plt.xlabel('date', fontsize = 20 )
    plt.title('Rolling Correlation Coefficient with %s (window: %d)' % (column, win), fontsize = 25 )
    plt.show()

    return rolling_coef.iloc[:, 1:].dropna()


def LinReg(X, y, return_value='coef'):
    df = pd.concat([X, y], axis=1).dropna()
    est = sm.OLS(df.iloc[:, 1], df.iloc[:, 0], missing='drop')
    result = est.fit()
    if return_value == 'coef':
        return result.params[0]
    elif return_value == 'p':
        return result.pvalues[0]
    elif return_value == 't':
        return result.tvalues[0]


def CAPM(return_data, market_column):
    y = return_data[market_column]
    result = return_data[ticker_list[1:]].apply(lambda x: LinReg(x, y))
    return pd.DataFrame(result, columns=['beta'])


def rolling_CAPM(return_data, win, market_column):
    x = return_data[market_column]
    rolling_beta = return_data[ticker_list[1:]].rolling(win).apply(
        lambda y: LinReg(x, y), raw=False).dropna()

    fig = plt.figure(dpi=120)
    rolling_beta.plot(figsize=(10, 6), ax=plt.gca())
    plt.ylabel('beta', fontsize = 20 )
    plt.xlabel('date', fontsize = 20 )
    plt.title('Rolling Beta (window: %d)' % win, fontsize = 25 )
    plt.show()


    return rolling_beta


def auto_reg(return_data):
    ar_coeff = return_data.apply(lambda x: LinReg(x, x.shift(1)), raw=False)
    ar_pvalue = return_data.apply(
        lambda x: LinReg(
            x,
            x.shift(1),
            return_value='p'),
        raw=False)
    ar_df = pd.concat([ar_coeff, ar_pvalue], axis=1)
    ar_df.columns = ['AR_coefficient', 'AR_pvalue']
    return ar_df


if __name__ == '__main__':
    ticker_list = [
        'SPY',
        'XLB',
        'XLE',
        'XLF',
        'XLI',
        'XLK',
        'XLP',
        'XLU',
        'XLV',
        'XLY']
    tickers = reduce(lambda x, y: x + y,
                     [ticker + ' ' for ticker in ticker_list])

    # 1. Download Ticker Data
    ETFs = ETF(tickers)
    print('a) Download historical price data of ETFs from January 1st 2010:')
    price_data = ETFs.get_price_data("2010-01-01", "2019-09-13")
    return_data = ETFs.cal_return_data()

    # 2. Annualized Return and Standard Deviation
    r, s = ETFs.cal_ann_return_std()
    return_std_df = pd.concat([r, s], axis=1)
    return_std_df.columns = ['Ann_Return', 'Standard_Deviation']
    print('b) Annualized Return and Standard Deviation of ETFs:')
    print(return_std_df.T)

    # 3. Covariance matrix of daily and monthly returns
    return_data_month = ETFs.cal_return_data(freq='M')
    corr_matrix_day = cov_matrix(return_data, 'corr')
    corr_matrix_month = cov_matrix(return_data_month, 'corr')
    diff_matrix = corr_matrix_day - corr_matrix_month
    print('c) Covariance Matrix:')
    print('Correlation Matrix of Daily Returns:')
    print(corr_matrix_day)
    print('Correlation Matrix of Monthly Returns:')
    print(corr_matrix_month)
    print('Difference of Monthly and Daily Correlation Matrix:')
    print(diff_matrix)

    # 4. Rolling Correlation
    print('d) Rolling Correlation with SPY:')
    rolling_coef = rolling_coef(return_data, 90, column='SPY')
    print(rolling_coef)

    # 5. Whole-period beta & Rolling beta
    print('e) CAPM model:')
    beta = CAPM(return_data, 'SPY')
    #####################################
    # rolling_beta = pd.read_csv('Rolling_beta.csv', index_col=0)
    rolling_beta = rolling_CAPM(return_data, 90, 'SPY')

    fig = plt.figure(dpi=120)
    rolling_beta.plot(figsize=(10, 6), ax=plt.gca())
    plt.ylabel('beta')
    plt.xlabel('date')
    plt.title('Rolling Beta (window: %d)' % 90)
    plt.show()
    #####################################
    print('Whole-peirod beta:')
    print(beta)
    print('Rolling beta with window of 90:')
    print(rolling_beta)

    # rolling_beta.to_csv('Rolling_beta.csv')

    # 6. Auto-Regression
    ar_coeff = auto_reg(return_data)
    print('f) AutoRegression Coefficient:')
    print(ar_coeff)
