from Option_Pricing import *
from scipy import stats
import statsmodels.api as sm

def cal_option_price(Option, steps, simu_times, option_type):
    simu_paths = Option.simulated_paths(steps, simu_times)
    S, *_ = simu_paths.generate_simu_paths('Bachelier', Option.S0, Option.vol, Option.r, plot=False) #seed =
    Option.payoff_simu(S, option_type, plot=False)
    simu_price = Option.simu_price()
    return simu_price


if __name__ ==  '__main__':
    K = 100; S0 = 100; r = 0; vol = 10; tau = 1
    steps = 250; simu_times = 1000

    # # a. Generate Paths
    lkbkOption = Lookback_option(K, S0, r, vol, tau)
    simu_paths = lkbkOption.simulated_paths(steps, simu_times)
    S, mean_terminal, var_terminal  = simu_paths.generate_simu_paths('Bachelier', S0, vol, r, seed=356)
    print(mean_terminal, var_terminal)


    # # b. terminal value histogram
    # plt.figure(figsize=[6, 3], dpi=120)
    # plt.hist(S[:,-1], rwidth=0.8)
    # plt.title('Distribution of Terminal Value of Asset')
    # plt.xlabel('Ending V')
    # plt.ylabel('Frequency')
    # # plt.show()
    #
    # ############ Distribution Test #############
    # # Distribution Test
    # # plt.figure(figsize=[10, 8], dpi=120)
    # # sorted_ = np.sort(S[:,-1])
    # # yvals = np.arange(simu_times) / float(simu_times)
    # # plt.plot(sorted_, yvals)
    # # plt.title('CDF of Ending Values')
    # # plt.xlabel('Ending Values')
    # # plt.ylabel('CDF')
    #
    # plt.figure(figsize=[10, 8], dpi=120)
    # sm.qqplot(S[:,-1], fit=True, line='45')
    # plt.title('QQplot', size = 25)
    # plt.xlabel('Theoretical Distribution', size = 20)
    # plt.ylabel('Sample Quantiles',size=20)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # ############################################
    #
    # c. Calculating Price
    simu_payoff, *_ = lkbkOption.payoff_simu(S, 'Put', plot=False)
    simu_price = lkbkOption.simu_price()
    print('Simu_price:',simu_price)
    plt.figure()
    plt.hist(simu_payoff)
    plt.show()


    # # d. Delta
    delta_dict = {}
    for e in np.linspace(0.01, 10, 300):
        option1 = Lookback_option(K, S0+e, r, vol, tau)
        option2 = Lookback_option(K, S0-e, r, vol, tau)

        price1 = cal_option_price(option1, steps, simu_times, 'Put')
        price2 = cal_option_price(option2, steps, simu_times, 'Put')

        # print(round(e,2),':',price1,price2)
        delta = (price1 - price2)/2/e
        delta_dict[e] = [delta, price1, price2]
    #
    # e_list = list(delta_dict.keys())
    # delta_list = [e[0] for e in list(delta_dict.values())]
    # plt.figure()
    # plt.plot(e_list, delta_list, linewidth=2.5)
    # plt.plot(np.linspace(-0.8,10.8,10),[-1]*10,ls='-.',linewidth = 3)
    # plt.title('Delta v.s \u0190',size=25)
    # plt.ylabel('delta',size=20)
    # plt.xlabel('\u0190',size=20)
    # plt.xticks(size=15)
    # plt.yticks(size=15)
    # plt.xlim(-0.2, 10.2)
    # plt.ylim(-2,0.5)
    # plt.show()
    #



