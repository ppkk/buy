# %%
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import numpy as np
from ipywidgets import interactive, fixed


#data = yf.download('AAPL', '2005-01-01', '2011-01-01')
data = yf.download('AAPL', '2006-01-01', '2010-01-01')

# %%
y = data.Open.to_numpy()
x = np.arange(0, len(y), 1)

# def indicator(der1, der2):
#     if 

# %%
def compute(smooth = 40, c_dx = 0, c_ddx = 0, c_sign_dx = 0, c_sign_ddx = 1):
    # cs=CubicSpline(x,y)
    uvs = UnivariateSpline(x, y)
    uvs.set_smoothing_factor(smooth)
    spl_x = uvs(x)
    spl_dx = uvs(x, 1)
    spl_ddx = uvs(x, 2)

    # normalize derivatives for the indicator calculation
    norm_dx = spl_dx/max(np.max(spl_dx), -np.min(spl_dx))
    norm_ddx = spl_ddx/max(np.max(spl_ddx), -np.min(spl_ddx))
    indicator = c_dx * norm_dx + c_ddx * norm_ddx + c_sign_dx * np.sign(norm_dx) + c_sign_ddx * np.sign(norm_ddx)

    # normalize indicator such that its minimum is 0 and sums to the number of days
    buy = indicator - np.min(indicator) 
    buy *= len(x) / np.sum(buy)
    
    # buy 1 stock each day
    cost_reference = sum(y)
    # buy based on indicator
    cost_indicator = np.dot(buy, y)

    fig, (val, buys, ind, der1, der2) = plt.subplots(figsize=(10, 8), nrows=5, height_ratios=[6, 3, 1, 1, 1])
    
    val.plot(x, y, label='data')
    val.plot(x, spl_x, label='S')
    val.grid()
    val.legend(loc='lower right')
    
    buys.plot(x, buy, label="buy ind, tot: " + str(cost_indicator), color="red")
    buys.plot(x, np.ones(len(x)), label="buy ref, tot: " + str(cost_reference), color="green")
    buys.grid()
    buys.legend(loc='upper right')

    ind.plot(x, indicator, label="ind", color="cyan")
    ind.grid()
    ind.legend(loc='lower right')
    
    der1.plot(x, norm_dx, label="norm dx", color="blue")
    der1.grid()
    der1.legend(loc='lower right')
    
    der2.plot(x, norm_ddx, label="norm dxx", color="green")
    der2.grid()
    der2.legend(loc='lower right')

# %%
compute()
# %%
w=interactive(compute,smooth=(0.0,200.0), c_dx=(-30,30), c_ddx=(-30,30), c_sign_dx=(-30,30), c_sign_ddx=(-30,30))
w

# %%
