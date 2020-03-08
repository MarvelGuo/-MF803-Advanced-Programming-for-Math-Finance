from ETFs import ETF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)


def eigen_decomposition(matrix, title=None, plot=False, print_out=False):
    m = np.array(matrix)
    eigen_value = LA.eig(m)[0]
    N = len(eigen_value)

    pos = len(eigen_value[eigen_value > 0]) / N
    neg = len(eigen_value[eigen_value < 0]) / N
    zero = len(eigen_value[eigen_value == 0]) / N

    if print_out:
        print('\n%d%% of eigenvalues are positive.' % (pos * 100))
        print('%d%% of eigenvalues are negative.' % (neg * 100))
        print('%d%% of eigenvalues are zero.' % (zero * 100))

    if plot:
        plt.figure()
        plt.plot(range(N), sorted(eigen_value, reverse=True))
        plt.title(title, fontsize=15)

    return pos, neg, zero


def regularized_cov_matrix(delta, matrix):
    matrix = np.array(matrix)
    return delta * np.diag(matrix.diagonal()) + (1 - delta) * matrix


def unconstraint_optm_weight(
        cov_matrix, expect_return, aversion, standarlized=False):
    raw_w = 1 / (2 * aversion) * \
        LA.inv(cov_matrix).dot(np.array(expect_return))
    if standarlized:
        return raw_w / np.sum(raw_w)
    else:
        return raw_w


def weight_instability(new_w, original_w):
    '''
    calculate the percentage change of new weight to the original weight
    original_w: array
    new_w:
    '''
    orig_w = original_w.reshape((1, -1))
    df_diff = (new_w - orig_w) / orig_w
    return df_diff


if __name__ == '__main__':
    ticker_list = [
        'XLB',
        'XLE',
        'XLF',
        'XLI',
        'XLK',
        'XLP',
        'XLU',
        'XLV',
        'XLY']
    tickers = ' '.join(ticker_list)
    start_date = '2010-01-01'
    end_date = '2019-11-21'
    ETFs = ETF(tickers)

    # problem 1.a
    ETFs.get_price_data(start_date, end_date)
    # ETFs.price_data.to_csv('price_data.csv')
    # ETFs.price_data = pd.read_csv('price_data.csv', index_col=0)

    # problem 1.b
    return_data = ETFs.cal_return_data()
    cov_m = ETFs.cov_matrix(return_data, 'cov')
    print('\ncovariance matrix:')
    print(cov_m)

    # problem 1.c
    title_1c = 'Eigenvalues of ETF Covariance Matrix'
    eigen_decomposition(cov_m, title=title_1c, plot=True, print_out=True)

    # problem 1.d
    # np.random.seed(1)
    rdm_X = np.random.normal(size=(len(cov_m), len(cov_m)))
    rdm_m = np.triu(rdm_X)
    rdm_m += rdm_m.T - np.diag(rdm_m.diagonal())

    # problem 1.e
    title_1e = 'Eigenvalues of random Covariance Matrix'
    eigen_decomposition(rdm_m, title=title_1e, plot=True, print_out=True)

    # problem 2.a
    ann_return, _ = ETFs.cal_ann_return_std()
    print('\nAnnualized Return:\n', ann_return)

    # problem 2.b
    orig_w = unconstraint_optm_weight(cov_m, ann_return, aversion=1)
    print(orig_w)

    # problem 2.c
    result = []
    sigma_list = [0.005, 0.01, 0.05, 0.1]
    for sigma in sigma_list:
        Z = np.random.normal(size=len(cov_m))
        Er = ann_return + Z * sigma
        w = unconstraint_optm_weight(cov_m, Er, aversion=1)
        result.append(w)
    df_c = pd.DataFrame(result, columns=ticker_list, index=sigma_list)

    df_diff_c = weight_instability(df_c, orig_w)
    df_diff_c.plot()
    plt.title('Percentage Change of new w from original w', fontsize=15)
    plt.xlabel('sigma', fontsize=10)
    plt.ylabel('Percentage Change', fontsize=10)

    # problem 2.e
    rglzed_cov_m = regularized_cov_matrix(delta=1, matrix=cov_m)
    title_2e = 'Eigenvalues of Regularized cov matrix with delta = 1'
    eigen_decomposition(
        rglzed_cov_m,
        title=title_2e, plot=True)
    print(
        '\nRank of Regularized Covariance Matrix:',
        LA.matrix_rank(rglzed_cov_m))

    # problem 2.f
    eig_result = []
    for delta in np.linspace(0, 1, 51):
        rglzed_cov_m = regularized_cov_matrix(delta, cov_m)
        result = eigen_decomposition(rglzed_cov_m)
        eig_result.append(result)

    eig_df = pd.DataFrame(eig_result)
    eig_df.index = np.linspace(0, 1, 51)
    eig_df.columns = ['positive %', 'negative %', 'zero %']
    print(eig_df)

    eig_df.plot()
    plt.title('Eigenvalues with different delta', fontsize=15)
    plt.xlabel('delta', fontsize=10)

    # problem 2.g
    df_list = []
    for sigma in sigma_list:
        Z = np.random.normal(size=len(cov_m))
        Er = ann_return + Z * sigma
        result = []
        for delta in np.linspace(0, 1, 6):
            rglzed_cov_m = regularized_cov_matrix(delta, cov_m)
            w = unconstraint_optm_weight(rglzed_cov_m, Er, aversion=1)
            result.append(w)
        df = pd.DataFrame(
            result,
            columns=ticker_list,
            index=np.linspace(
                0,
                1,
                6))
        df_list.append(df)

    plt.figure()
    for i in range(len(df_list)):
        plt.subplot(2, 2, i + 1)
        diff_i = weight_instability(df_list[i], orig_w)
        diff_i.plot(ax=plt.gca())
        plt.title('sigma = %s' % sigma_list[i], fontsize=14)
        plt.xlabel('delta', fontsize=12)
        plt.ylabel('Percentage Change', fontsize=12)
        plt.legend(fontsize=6, ncol=3)

    plt.tight_layout()
    plt.show()
