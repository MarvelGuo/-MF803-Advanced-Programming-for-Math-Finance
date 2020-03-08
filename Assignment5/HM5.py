import json
from Yield_Curve_revised import *
import numpy as np
import pandas as pd
import copy
from numpy import exp
import matplotlib.pyplot as plt
plt.style.use('ggplot')


years = [1, 2, 3, 4, 5, 7, 10, 30]
swap_rate = [2.8438, 3.06, 3.126, 3.144, 3.15, 3.169, 3.21, 3.237]
swap_rate = list(map(lambda x: x / 100, swap_rate))
swap = dict(zip(years, swap_rate))

# problem a-c
print('\nproblem a-c')
Yield = Yield_Curve(swap)
print(json.dumps(Yield.FwR, sort_keys=True, indent=4, separators=(',', ': ')))

plt.figure(dpi=120)
plt.plot(years, list(Yield.FwR.values()))
plt.plot(years, list(swap.values()))
plt.title('forward rates vs. swap rates')
plt.legend(['Forward Rates', 'Swap Rates'])
plt.xlabel('Maturity')
plt.ylabel('Rates')

# problem d
print('\nproblem d')
print('breakeven swap rate of a 15Y swap is:', Yield.breakeven_swap(15))

# problem e
print('\nproblem e')
df = pd.DataFrame([Yield.Discount, Yield.ZeroRates]).T
df.columns = ['Discount Factor', 'Zero Rates']
df.index = np.linspace(0.5, 30, 60)
print('Head of Discount Factor & Zero Rates:')
print(df.head())
print('Tail of Discount Factor & Zero Rates:')
print(df.tail())

plt.figure(dpi=120)
plt.plot(np.linspace(0.5, 30, 60), Yield.ZeroRates)
plt.plot(years, list(swap.values()))
plt.title('Zero Rates vs. Swap Rates')
plt.legend(['Zero Rates', 'Swap Rates'])
plt.xlabel('Maturity')
plt.ylabel('Rates')

# problem f
# when forward rate changes, discount rate changes
Yield_f = copy.copy(Yield)
Yield_f.FwR = dict([(y, Yield.FwR[y] + 0.01) for y in Yield.FwR])
Yield_f.Discount = [d * exp(0.01 / 2 * i)
                    for i, d in enumerate(Yield.Discount, start=1)]
new_swap = []
for y in years:
    temp_swap = Yield_f.breakeven_swap(y)
    new_swap.append(temp_swap)
Yield_f.FwR = dict(zip(years, new_swap))

df = pd.DataFrame([swap_rate, new_swap]).T
df.columns = ['Old Swap (%)', 'New Swap (%)']
df['Old Swap (%)'] *= 100
df['New Swap (%)'] *= 100
df['difference (bp)'] = (df['New Swap (%)'] - df['Old Swap (%)']) * 100
df.index = years

print('\nproblem f')
print(df)

# problem g - h
swap_rate_g = [
    2.8438,
    3.06,
    3.126,
    3.144 + 0.05,
    3.15 + 0.1,
    3.169 + 0.15,
    3.21 + 0.25,
    3.237 + 0.5]
swap_rate_g = list(map(lambda x: x / 100, swap_rate_g))
swap_g = dict(zip(years, swap_rate_g))
Yield_g = Yield_Curve(swap_g)

print('\nproblem g-h')
print(
    json.dumps(
        Yield_g.FwR,
        sort_keys=True,
        indent=4,
        separators=(
            ',',
            ': ')))


plt.figure(dpi=120)
plt.plot(years, list(Yield.FwR.values()))
plt.plot(years, list(Yield_g.FwR.values()))
plt.title('Bearish Steepener of Swap Rates')
plt.legend(['Original Forward Rates', 'Bearish Forward Rates'])
plt.xlabel('Maturity')
plt.ylabel('Rates')

# problem i - j
swap_rate_i = [
    2.8438 - 0.5,
    3.06 - 0.25,
    3.126 - 0.15,
    3.144 - 0.1,
    3.15 - 0.05,
    3.169,
    3.21,
    3.237]
swap_rate_i = list(map(lambda x: x / 100, swap_rate_i))
swap_i = dict(zip(years, swap_rate_i))
Yield_i = Yield_Curve(swap_i)

print('\nproblem i-j')
print(
    json.dumps(
        Yield_i.FwR,
        sort_keys=True,
        indent=4,
        separators=(
            ',',
            ': ')))


plt.figure(dpi=120)
plt.plot(years, list(Yield.FwR.values()))
plt.plot(years, list(Yield_i.FwR.values()))
plt.title('Bull Steepener of Swap Rates')
plt.legend(['Original Forward Rates', 'Bull Forward Rates'])
plt.xlabel('Maturity')
plt.ylabel('Rates')

plt.show()
