#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('ggplot')


class Simulation():
    def __init__(self, ):



class Option():
    def __init__(self, K, S0, r, vol):
        self.K = K
        self.S0 = S0
        self.r = r
        self.vol = vol

    def Simulation(self, normal_mean, steps,
                   simu_times, plot=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        random_num = np.random.normal(
            normal_mean, 1 / 252 ** 0.5, [simu_times, steps])
        move = random_num * self.vol + self.r * 1 / 365

        S = np.zeros([simu_times, steps])
        S[:, 0] = self.S0
        for t in range(1, steps):
            S[:, t] = S[:, t - 1] + S[:, t - 1] * move[:, t - 1]

        if plot:
            plt.figure(figsize=(8, 4), dpi=120)
            plt.plot(S.T, linewidth=1)
            plt.title('Simulated Paths of Security Price')
            plt.xlabel('steps')
            plt.ylabel('Price')
            # plt.show()

        mean_terminal = S.mean(axis=0)[-1]
        var_terminal = S.var(axis=0)[-1]

        return S, mean_terminal, var_terminal

    def discount(self, x, year):
        return x * np.exp(- self.r * year)

    def simu_price(self, payoff, T):
        simu_price = self.discount(payoff, T)
        return simu_price



class Euro_option(Option):
    def __init__(self, K, S0, r, vol):
        Option.__init__(self, K, S0, r, vol)

    def payoff_simu_put(self, S, plot=True):
        pay = K - S[:, -1]
        temp_pay = map(lambda x: max(x, 0), pay)
        payoff = np.array(list(temp_pay))

        if plot:
            plt.figure(figsize=[6, 3], dpi=120)
            plt.hist(payoff, rwidth=0.8)
            plt.title('Distribution of Payoff of European option')
            plt.xlabel('Payoff')
            plt.ylabel('Frequency')
            # plt.show()

        pay_mean = payoff.mean()
        pay_std = payoff.std()
        return pay_mean, pay_std, payoff

    def BSM(self, T):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.vol ** 2 / 2)
              * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)

        C = S0 * N_d1 - self.K * np.exp(-self.r * T) * N_d2
        P = C + self.K * np.exp(-self.r * T) - self.S0

        return C, P


class Lookback_option(Option):
    def __init__(self, K, S0, r, vol):
        self.K = K
        self.S0 = S0
        self.r = r
        self.vol = vol

    def payoff_simu_put(self, S, plot=True):
        pay = self.K - S.min(axis=1)
        temp_pay = map(lambda x: max(x, 0), pay)
        payoff = np.array(list(temp_pay))

        if plot:
            plt.figure(figsize=[6, 3], dpi=120)
            #     plt.hist(payoff, bins = int(S.shape[0]**0.5),density=True, color = 'green',range = (0, (max(payoff)+1)//1))
            plt.hist(payoff, rwidth=0.8)
            plt.title('Distribution of Payoff of Lookback put option')
            plt.xlabel('Payoff')
            plt.ylabel('Frequency')
            # plt.show()

        pay_mean = payoff.mean()
        pay_std = payoff.std()
        return pay_mean, pay_std, payoff


if __name__ == '__main__':

    K = 100
    S0 = 100
    r = 0
    vol = 0.25

    steps = 250
    simu_times = 1000

    normal_mean = 0

    # a. Generate simulated paths
    Europe_Option_pricing = Euro_option(K, S0, r, vol)
    S, mean_terminal, var_terminal = Europe_Option_pricing.Simulation(
        normal_mean, steps, simu_times, seed = 20)  # , seed=20
    print('(a) Generate simulated paths')
    print('The mean of the terminal value of these paths is:', mean_terminal)
    print('The variance of the terminal value of these paths is:', var_terminal)

    # b. Calculate the simulated payoff of Euro put option
    pay_mean, pay_std, payoff = Europe_Option_pricing.payoff_simu_put(S)
    print('\n(b) Calculate the simulated payoff of Euro put option')
    print('The mean of the payoff:', pay_mean)
    print('The standard deviation of the payoff:', pay_std)

    # c. Calculate the simulated price of European put option
    T = steps / 252
    simu_price_put = Europe_Option_pricing.simu_price(pay_mean, T)
    print('\n(c) Calculate the simulated price of European put option')
    print(
        'Simulated approximation to the price of European put option is:',
        simu_price_put)

    # d. Compare the simulated price of European Option with BSM price
    C, P = Europe_Option_pricing.BSM(T)
    print('\n(d) Compare the simulated price of European Option with BSM price')
    print('price of European Option calculated by BSM is:', P)
    print('Difference of the two types of prices:', abs(P - simu_price_put))

    # e. Calculate the simulated payoff and price of fixed strike lookback put
    # option
    Lookback_option_pricing = Lookback_option(K, S0, r, vol)
    pay_lookback_mean, pay_lookback_std, lookback_payoff = Lookback_option_pricing.payoff_simu_put(
        S)
    simu_price_lookback_put = Lookback_option_pricing.simu_price(
        pay_lookback_mean, T)
    print('\n(e) Calculate the simulated payoff of fixed strike lookback put option')
    print(
        'the simulated payoff of fixed strike lookback put option:',
        pay_lookback_mean)
    print(
        'the simulated price of fixed strike lookback put option:',
        simu_price_lookback_put)

    # f. Calculate the premium for the extra optionality embedded in the
    # lookback option
    premium = simu_price_lookback_put - simu_price_put
    print('\n(f) Calculate the premium for the extra optionality embedded in the lookback option')
    print('The premium is:', premium)

    # g. Price variance with different volatility
    price_dict = {}
    for v in np.linspace(0.1, 0.4, 31):
        euro_option = Euro_option(K, S0, r, v)
        lookback_option = Lookback_option(K, S0, r, v)
        S_chg, *_ = euro_option.Simulation(normal_mean, steps, simu_times, plot=False)

        euro_payoff_mean, *_ = euro_option.payoff_simu_put(S_chg, plot=False)
        lookback_payoff_mean, *_ = lookback_option.payoff_simu_put(S_chg, plot=False)

        euro_option_price = euro_option.simu_price(euro_payoff_mean, T)
        lookback_option_price = lookback_option.simu_price(
            lookback_payoff_mean, T)
        premium = lookback_option_price - euro_option_price

        price_dict[v] = [euro_option_price, lookback_option_price, premium]

    price_df = pd.DataFrame.from_dict(
        price_dict, orient='index', columns=[
            'euro', 'lookback', 'premium'])

    print('\n(e) Price of options with different volatility')
    print(price_df)
    price_df.iloc[:, :2].plot(linewidth=4.5)
    plt.legend(loc='upper left', fontsize=20)
    plt.xlabel('volatility', fontsize=20)
    plt.ylabel('price', fontsize=20)
    plt.title('Price of options with different volatility', fontsize=25)
    plt.fill_between(
        np.linspace(
            0.1,
            0.4,
            31),
        price_df.euro,
        price_df.lookback,
        alpha=0.2,
        color='yellow')
    plt.show()
