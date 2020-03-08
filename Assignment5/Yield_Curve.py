import numpy as np
from scipy.optimize import fsolve
from numpy import exp


class Yield_Curve:
    def __init__(self, swap):
        self.swap = swap
        self.years = list(swap.keys())
        self.FwR = {}
        self.Discount = None  # List of Discount factors; every half year
        self.cal_FwR()
        self.ZeroRates = self.cal_ZeroRates()

    def pricing_func(self, rf, D, T, diff_T):
        # new discount factors
        D_new = 0
        for i in np.linspace(0.5, diff_T, diff_T * 2):
            last_D = 1 if len(D) == 0 else D[-1]
            temp_D = last_D * exp(-i * rf)
            D_new += temp_D

        # old float legs is same as the fixed legs of previous period
        float_legs = self.swap[T - diff_T] * (sum(D)) * 0.5 if T != 1 else 0
        float_new = rf / 2 * D_new

        C = self.swap[T]
        fixed_pv = C * (sum(D) + D_new) * 0.5
        float_pv = float_legs + float_new
        func = fixed_pv - float_pv
        return func

    def cal_FwR(self):
        D = []
        # Calculate Forward Rate
        for i in range(0, len(self.swap)):
            T = self.years[i]
            # difference between the given year and previous given year
            diff_T = T - self.years[i - 1] if i != 0 else 1
            sol = fsolve(self.pricing_func, 0.03, args=(D, T, diff_T))[0]
            self.FwR[T] = sol

            # Update Discount Factor
            D_new = []
            for i in np.linspace(0.5, diff_T, diff_T * 2):
                last_D = 1 if len(D) == 0 else D[-1]
                temp_D = last_D * exp(-i * self.FwR[T])
                D_new.append(temp_D)
            D.extend(list(D_new))
        self.Discount = D

    def cal_ZeroRates(self):
        return [-np.log(d) * 2 / (i + 1) for i, d in enumerate(self.Discount)]

    def breakeven_swap(self, Y):
        swap_years = np.array(self.years)

        fixed_legs = 0.5 * sum(self.Discount[:Y * 2])
        float_legs = 0
        for y in np.linspace(0.5, Y, 2 * Y):
            forward_year = swap_years[swap_years >= y][0]
            float_legs += 0.5 * \
                self.FwR[forward_year] * self.Discount[int(2 * y - 1)]

        bkev_swap = float_legs / fixed_legs
        return bkev_swap


if __name__ == '__main__':
    years = [1, 2, 3, 4, 5, 7, 10, 30]
    swap_rate = [2.8438, 3.06, 3.126, 3.144, 3.15, 3.169, 3.21, 3.237]
    swap_rate = list(map(lambda x: x / 100, swap_rate))
    swap = dict(zip(years, swap_rate))

    Yield = Yield_Curve(swap)
    Yield.cal_FwR()
    print(Yield.FwR)
