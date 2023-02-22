import numpy as np
from math import factorial
import matplotlib.pyplot as plt
    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

samp_rates = np.logspace(-3,4,1000, base=2)

fig,axs  = plt.subplots(1,2)
plt.suptitle('Single Experiment Smooth Curved')

data = np.load('curves.npy')

data = data[0]
datah = savitzky_golay(data, 21, 3)
axs[0].set_title('Non-Logged Hit-Rates')
axs[0].plot(samp_rates, data, c='gray',label='Hit-Rates',linewidth=1,alpha=0.2)
axs[0].plot(samp_rates, datah, c='g',label='Savitzky-Smoothed Data')
axs[0].set_xscale('log',base=2)
axs[0].legend()

ldata = (-1)*np.log(data)
ldatah = savitzky_golay(ldata, 21, 3)
axs[1].set_title('Neg-Logged Hit-Rates')
axs[1].plot(samp_rates,ldata,c='gray',label='Logged-Hit-Rates',linewidth=1,alpha=0.2)
axs[1].plot(samp_rates,ldatah,c='g',label='Savitzky-Smoothed Logged-Hit-Rates')
axs[1].set_xscale('log',base=2)
axs[1].legend()

plt.show()

