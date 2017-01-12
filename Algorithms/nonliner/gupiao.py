import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt

stock = ['002697.XSHE','600783.XSHG','000413.XSHE','601588.XSHG']
start_date = '2015-01-01'
end_date = '2015-12-31'
df = get_price(stock, start_date, end_date, 'daily',['close'])
data = df['close']
data.head()
