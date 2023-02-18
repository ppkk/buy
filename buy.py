# %%
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import numpy as np
from ipywidgets import interactive, fixed


#data = yf.download('AAPL', '2005-01-01', '2011-01-01')
data = yf.download('AAPL', '2006-01-01', '2010-01-01')
#data = yf.download('AAPL', '2006-01-01', '2006-03-20')

# %%
y = data.Open.to_numpy()
x = np.arange(0, len(y), 1)

# def indicator(der1, der2):
#     if 

# %%
def predict(y, smooth):
    x = np.arange(0, len(y), 1)
    uvs = UnivariateSpline(x, y)
    uvs.set_smoothing_factor(smooth)
    spl_val = uvs(x)
    spl_dx = uvs(x, 1)
    spl_ddx = uvs(x, 2)
    return spl_val[-1], spl_dx[-1], spl_ddx[-1]

u, d, dd = predict(y[0:10], 40)
print(y[0:10], u, d, dd)


def create_prediction(y, smooth):
    pred = np.zeros(len(y))
    pred_dx = np.zeros(len(y))
    pred_ddx = np.zeros(len(y))
    pred[:min(len(y), 3)]
    for i in range(3, len(y)):
        pred[i], pred_dx[i], pred_ddx[i] = predict(y[:i+1], smooth)

    return pred, pred_dx, pred_ddx

def normalize_derivatives(dx, ddx): 
    # normalize derivatives for the indicator calculation
    norm_dx = dx/max(np.max(dx), -np.min(dx))
    norm_ddx = ddx/max(np.max(ddx), -np.min(ddx))

    return norm_dx, norm_ddx

def compute_indicator(norm_dx, norm_ddx, c_dx, c_ddx, c_sign_dx, c_sign_ddx):
    indicator = c_dx * norm_dx + c_ddx * norm_ddx + c_sign_dx * np.sign(norm_dx) + c_sign_ddx * np.sign(norm_ddx)

    # normalize indicator such that its minimum is 0 and sums to the number of days
    buy = indicator - np.min(indicator) 
    buy *= len(x) / np.sum(buy)

    return indicator, buy

# %%
def compute(smooth = 40, c_dx = 0, c_ddx = 0, c_sign_dx = 0, c_sign_ddx = 1):
    # cs=CubicSpline(x,y)
    uvs = UnivariateSpline(x, y)
    uvs.set_smoothing_factor(smooth)
    spl_val = uvs(x)
    spl_dx = uvs(x, 1)
    spl_ddx = uvs(x, 2)

    norm_dx, norm_ddx = normalize_derivatives(spl_dx, spl_ddx)
    indicator, buy = compute_indicator(norm_dx, norm_ddx, c_dx, c_ddx, c_sign_dx, c_sign_ddx)
    
    pred_val, pred_dx, pred_ddx = create_prediction(y, smooth)
    pred_norm_dx, pred_norm_ddx = normalize_derivatives(pred_dx, pred_ddx)
    pred_indicator, pred_buy = compute_indicator(pred_norm_dx, pred_norm_ddx, c_dx, c_ddx, c_sign_dx, c_sign_ddx)

    # buy 1 stock each day
    cost_reference = sum(y)
    # buy based on indicator
    cost_indicator = np.dot(buy, y)

    pred_cost_indicator = np.dot(pred_buy, y)

    fig, (val, buys, ind, der1, der2) = plt.subplots(figsize=(10, 8), nrows=5, height_ratios=[6, 3, 1, 1, 1])
    
    val.plot(x, y, label='data', color='blue')
    val.plot(x, spl_val, label='interpolate', color='red')
    val.plot(x, pred_val, label='predict', color='green')
    val.grid()
    val.legend(loc='lower right')
    
    buys.plot(x, np.ones(len(x)), label="buy ref, tot: " + str(cost_reference), color="blue")
    buys.plot(x, buy, label="buy ind, tot: " + str(cost_indicator), color="red")
    buys.plot(x, pred_buy, label="buy pred ind, tot: " + str(pred_cost_indicator), color="green")
    buys.grid()
    buys.legend(loc='upper right')

    ind.plot(x, indicator, label="ind", color="magenta")
    ind.plot(x, pred_indicator, label="pred ind", color="cyan")
    ind.grid()
    ind.legend(loc='lower right')
    
    der1.plot(x, norm_dx, label="norm dx", color="red")
    der1.plot(x, pred_norm_dx, label="pred norm dx", color="green")
    der1.grid()
    der1.legend(loc='lower right')
    
    der2.plot(x, norm_ddx, label="norm dxx", color="red")
    der2.plot(x, pred_norm_ddx, label="pred norm dxx", color="green")
    der2.grid()
    der2.legend(loc='lower right')

# %%
compute()
# %%
w=interactive(compute,smooth=(0.0,200.0), c_dx=(-30,30), c_ddx=(-30,30), c_sign_dx=(-30,30), c_sign_ddx=(-30,30))
w

# %%
