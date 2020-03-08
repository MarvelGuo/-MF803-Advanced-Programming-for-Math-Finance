import pandas as pd
import statsmodels.api as sm
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from scipy.stats import kstest
from ETF import *
from pyfinance.ols import PandasRollingOLS

plt.style.use('ggplot')
# display all columns
pd.set_option('display.max_columns', None)


def rolling_coef(data, win):
    '''
    :param return_data:
    :param win: length of rolling window
    :param column: which column is calculated with
    '''
    rolling_coef = pd.DataFrame()
    pair = [(0, 1), (0, 2), (1, 2)]
    for i in pair:
        temp = data.iloc[:, i[0]].rolling(90).corr(data.iloc[:, i[1]])
        temp = pd.DataFrame(temp, columns=[data.columns[i[0]] + '_' + data.columns[i[1]]])
        rolling_coef = pd.concat([rolling_coef, temp], axis=1)

    colors = iter(['tomato', 'deepskyblue', 'orchid'])
    for c in rolling_coef.columns:
        fig = plt.figure(dpi=100)
        rolling_coef[c].plot(figsize=(25, 8), color=next(colors))
        plt.ylabel('Coefficient', fontsize=20)
        plt.xlabel('date', fontsize=20)
        plt.title('Rolling Correlation Coefficient between %s(window: %d)' % (c, win), fontsize=25)

    # plt.show()
    return rolling_coef.dropna()

def LinReg(X, y, return_value):
    df = pd.merge(X, y, left_index=True, right_index=True)
    est = sm.OLS(df.iloc[:,-1], df.iloc[:,:3], missing='drop')
    result = est.fit()
    if return_value == 'beta':
        return result.params
    elif return_value == 'residual':
        return result.resid #y - est.predict()


if __name__ == '__main__':
    # f = open(r'C:\Users\PC\Desktop\2019fall\803\HM\HM2\F-F_Research_Data_Factors_daily.csv', 'r')
    f = open(r'F-F_Research_Data_Factors_daily.csv', 'r')
    data = pd.read_csv(f, index_col=0)

    factor = data.iloc[:, :3]
    factor.index = pd.to_datetime(factor.index, format="%Y%m%d")
    factor = factor[factor.index >= '2010-01-01']
    factor.columns = ['MKT', 'SMB', 'HML']

    # b.
    daily_cov = factor.cov()
    print('daily covariance matrix of factor data: ')
    print(daily_cov)
    daily_corr = factor.corr()
    print('daily correlation matrix of factor data: ')
    print(round(daily_corr),2)

    # c.
    rolling_coef = rolling_coef(factor, 90)
    print(rolling_coef)

    # d. distribution test
    test_result = {}
    for c in factor.columns:
        test_stat = kstest(factor[c], 'norm')
        test_result[c] = test_stat.pvalue
    print('The result of KS test')
    print(round(pd.DataFrame.from_dict(test_stat),2))

    # e. Beta
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
    ETFs = ETF(tickers)
    price_data = ETFs.get_price_data("2010-01-01", "2019-08-01")
    ETF_return = ETFs.cal_return_data()

    beta = ETF_return.apply(lambda x: LinReg(factor,x, return_value='beta'))
    print(beta.head)

    beta_result = {}
    plt.figure(figsize = (15,40),dpi=120)
    for i, c in enumerate(ETF_return):
        plt.subplot(5,2,i+1)
        model = PandasRollingOLS(y=ETF_return[c], x=factor, window=90)
        beta_result[c] = model.beta
        model.beta.plot(ax = plt.gca())


    # f. residual
    residual_dict = {}
    residual_test = {}
    for c in ETF_return:
        residual_dict[c] = LinReg(ETF_return[c], factor, return_value = 'residual')
        test_stat = kstest(residual_dict[c], 'norm')
        residual_test[c] = test_stat.pvalue

    for k, v in residual_test.items():
        print('symbol: {}, p_value: {}'.format(k, v))


    plt.show()





