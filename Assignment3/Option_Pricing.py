#!/usr/bin/env python
# coding: utf-8
import sys
from os import path
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('ggplot')


class Simulation():
    def __init__(self, steps, simu_times):
        self.steps = steps
        self.simu_times = simu_times

    def generate_simu_paths(self, model, S0, vol, r,
                            plot=True, seed=None):

        if model == 'BSM':
            S = self.Black_Scholes_path(S0, vol, r, seed)
        elif model == 'Bachelier':
            S = self.Bachelier_path(S0, vol, r, seed)
        else:
            raise ValueError('No such Simulated Model!!')

        if plot:
            self.plot_path(S, model)
            # plt.show()

        mean_terminal = S.mean(axis=0)[-1]
        var_terminal = S.var(axis=0)[-1]

        return S, mean_terminal, var_terminal

    def Black_Scholes_path(self, S0, vol, r, seed):
        if not seed:
            np.random.seed(seed)
        random_num = np.random.normal(
            0, 1 / 252 ** 0.5, [self.simu_times, self.steps])
        move = random_num * vol + r * 1 / 252

        S = np.zeros([self.simu_times, self.steps])
        S[:, 0] = S0
        for t in range(1, steps):
            S[:, t] = S[:, t - 1] + S[:, t - 1] * move[:, t - 1]

        return S

    def Bachelier_path(self, S0, vol, r, seed):
        if seed is not None:
            np.random.seed(seed)
        random_num = np.random.normal(
            0, 1 / 252 ** 0.5, [self.simu_times, self.steps])
        move = random_num * vol + r * 1 / 252

        S = np.zeros([self.simu_times, self.steps])
        S[:, 0] = S0
        for t in range(1, self.steps):
            S[:, t] = S[:, t - 1] + move[:, t - 1]

        return S

    def plot_path(self, paths, path_name):
        plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(paths.T, linewidth=1)
        plt.title('Simulated Paths of Security Price (%s)' % path_name)
        plt.xlabel('steps')
        plt.ylabel('Price')


class Option():
    def __init__(self, K, S0, r, vol, tau):
        self.K = K
        self.S0 = S0
        self.r = r
        self.vol = vol
        self.tau = tau
        self.payoff = None

    def simulated_paths(self, steps, simu_times):
        self.simulation = Simulation(steps, simu_times)
        return self.simulation

    def discount(self, x):
        return x * np.exp(- self.r * self.tau)

    def simu_price(self):
        pay_mean = self.payoff.mean()
        simu_price = self.discount(pay_mean)
        return simu_price


class Euro_option(Option):
    def __init__(self, K, S0, r, vol, tau):
        super().__init__(K, S0, r, vol, tau)
        # Option.__init__(self, K, S0, r, vol)

    def payoff_simu(self, S, option_type, plot=True):
        if option_type == 'Put':
            pay = self.K - S[:, -1]
        elif option_type == 'Call':
            pay = S[:, -1] - self.K
        else:
            raise ValueError('Option Type Must be Put or Call')

        temp_pay = map(lambda x: max(x, 0), pay)
        self.payoff = np.array(list(temp_pay))  # final payoffs of all paths

        if plot:
            payoff_hist(self)

        pay_mean = self.payoff.mean()
        pay_std = self.payoff.std()
        return self.payoff, pay_mean, pay_std

    def BSM(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.vol ** 2 / 2)
              * self.tau) / (self.vol * np.sqrt(self.tau))
        d2 = d1 - self.vol * np.sqrt(self.tau)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)

        C = self.S0 * N_d1 - self.K * np.exp(-self.r * self.tau) * N_d2
        P = C + self.K * np.exp(-self.r * self.tau) - self.S0

        return C, P

    def payoff_hist(self):
        plt.figure(figsize=[6, 3], dpi=120)
        plt.hist(self.payoff, rwidth=0.8)
        plt.title('Distribution of Payoff of European option')
        plt.xlabel('Payoff')
        plt.ylabel('Frequency')


class Lookback_option(Option):
    def __init__(self, K, S0, r, vol, tau):
        super().__init__(K, S0, r, vol, tau)

    def payoff_simu(self, S, option_type, plot=True):
        if option_type == 'Put':
            pay = self.K - S.min(axis=1)
        elif option_type == 'Call':
            pay = S.max(axis=1) - self.K
        else:
            raise ValueError('Option Type Must be Put or Call!')

        temp_pay = map(lambda x: max(x, 0), pay)
        self.payoff = np.array(list(temp_pay))

        if plot:
            payoff_hist(self)

        pay_mean = self.payoff.mean()
        pay_std = self.payoff.std()
        return self.payoff, pay_mean, pay_std

    def payoff_hist(self):
        plt.figure(figsize=[6, 3], dpi=120)
        plt.hist(self.payoff, rwidth=0.8)
        plt.title('Distribution of Payoff of Lookback option')
        plt.xlabel('Payoff')
        plt.ylabel('Frequency')


if __name__ == '__main__':

    K = 100
    S0 = 100
    r = 0
    vol = 0.25

    steps = 250
    simu_times = 1000

    normal_mean = 0

    print(Euro_option(K,S0,r,vol, 20/250).BSM()[0])
