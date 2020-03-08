from ETF import *
from Option_Pricing import *

import statsmodels.api as sm
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

import numpy as np


def AR_test(data, nlags, title, plot=True):
    ar_coef, _, p = stattools.acf(data, fft=False, nlags=nlags, qstat=True)
    if plot:
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_acf(data, ax=ax)
        plt.title('Auto-correlation of ' + title, size=25)
        plt.xlabel('time_lags', size=20)
        plt.ylabel('AR coefficients', size=20)
        plt.xticks(size=18)
        plt.yticks(size=18)
    ar_df = pd.DataFrame(ar_coef, columns=['AR_coef'])
    p_df = pd.DataFrame(p, columns=['p_value'])
    return pd.concat([ar_df, p_df], axis=1)


if __name__ == '__main__':
    ## a
    start_date = '2005-01-01'
    end_date = '2019-09-01'
    SPY_C = ETF('SPY')
    VIX_C = ETF('^VIX')  # adding '^' !

    SPY_p = SPY_C.get_price_data(start_date, end_date)
    SPY_r = SPY_C.cal_return_data()
    VIX = VIX_C.get_price_data(start_date, end_date)

    ## b
    SPY_p_AR = AR_test(SPY_p, 20, 'SPY', plot=True)
    # SPY_r_AR = AR_test(SPY_r, 20, plot=True)
    VIX_AR = AR_test(VIX, 20, 'VIX', plot=True)
    print('\nAR test result for SPY:')
    print(SPY_p_AR.head(6))
    print('\nAR test result for VIX:')
    print(VIX_AR.head(6))
    # plt.show()

    # c
    daily_corr = SPY_p.corr(VIX)
    print('daily correlation of SPY and VIX:', daily_corr)

    SPY_month = SPY_C.cal_month_end()
    VIX_month = VIX_C.cal_month_end()
    month_corr = SPY_month.corr(VIX_month)
    print('monthly correlation of SPY and VIX:', month_corr)

    ## d
    price_data = pd.concat([SPY_p, VIX], axis=1)
    price_data.columns = ['SPY', 'VIX']
    rolling_coef = price_data.rolling(90).corr(price_data['VIX']).iloc[:, 0].dropna()
    print('\nRolling coeffeicient with window of 90:')
    print(rolling_coef.head())

    plt.figure()
    rolling_coef.plot()
    plt.plot(rolling_coef.index, [rolling_coef.mean()] * len(rolling_coef), linewidth=2, linestyle='-.')
    plt.title('Rolling coeffeicient of SPY and implied volatility', size=22)
    plt.xlabel('date', size=18)
    plt.ylabel('Correlation coefficients', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)

    ## e
    plt.show()
    SPY_rlzed_vol = SPY_C.cal_rlzed_vol()
    vol_data = pd.merge(SPY_rlzed_vol, VIX, left_index=True, right_index=True)
    vol_data.columns = ['rlzd', 'implied']
    vol_data['premium'] = vol_data.implied - vol_data.rlzd

    plt.figure()
    vol_data['premium'].plot()
    plt.title('Premium of volatility', size=25)
    plt.xlabel('date', size=20)
    plt.ylabel('Premium', size=20)
    plt.xticks(size=18)
    plt.yticks(size=18)

    ## f
    rf = 0;
    tau = 1 / 12
    straddle = pd.merge(SPY_p, VIX, left_index=True, right_index=True)
    straddle.columns = ['S0', 'vol']
    straddle['call'] = Euro_option(straddle['S0'], straddle['S0'], rf, straddle['vol'] / 100, tau).BSM()[0]
    straddle['put'] = Euro_option(straddle['S0'], straddle['S0'], rf, straddle['vol'] / 100, tau).BSM()[1]
    # print(straddle.head())

    ## g
    payoff = abs(straddle.S0 - straddle.S0.shift(21))
    payoff.name = 'payoff'
    straddle = pd.concat([straddle, payoff.shift(-21)], axis=1).dropna()
    straddle['p/l'] = straddle.payoff - straddle.call - straddle.put

    plt.figure()
    straddle['p/l'].plot()
    plt.title('Profit of Straddles', size=25)
    plt.xlabel('date', size=20)
    plt.ylabel('G/L', size=20)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.plot(straddle.index, [straddle['p/l'].mean()] * len(straddle), linestyle='-.', linewidth=3)
    print('\nAverage P/L:', straddle['p/l'].mean())

    ## h
    straddle = pd.merge(straddle, vol_data['premium'], left_index=True, right_index=True)
    plt.figure()
    plt.scatter(straddle.premium, straddle['p/l'])
    plt.title('P/l v.s. Premium', size=25)
    plt.xlabel('Premium of volatility', size=20)
    plt.ylabel('P/L', size=20)

    y = straddle['p/l']
    X = straddle.premium
    X = sm.add_constant(X)
    est = sm.OLS(y, X, missing='drop')
    result = est.fit()
    # print(result.summary())

    plt.plot(straddle.premium, result.fittedvalues, color='black')
    print(straddle)

    plt.show()
