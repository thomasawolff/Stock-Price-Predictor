from __future__ import division
import pandas as pd
from pandas import Series,DataFrame,datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from pandas_datareader import DataReader
from datetime import datetime

tech_list =['AAPL'] #,'GOOG','MSFT','AMZN','NVDA','FB']

end = datetime.now()
start = datetime(end.year-5,end.month,end.day)

for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)

closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close']
last_price = closing_df.iloc[-1].iloc[-1]
##print(closing_df)
tech_rets = closing_df.pct_change()
rets = tech_rets.dropna()
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
##NVDA['Daily Return'] = NVDA['Adj Close'].pct_change()
days = 365
dt = 1/days
mu = rets.mean()
sigma = rets.std()


def stock_monte_carlo(last_price,days,mu,sigma):
    price = np.zeros(days)
    price[0] = last_price

    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in range(1,days):
        shock[x] = np.random.normal(loc = mu*dt,scale = sigma*np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price


def montePlotter():
    for run in range(100):
        plt.plot(stock_monte_carlo(last_price,days,mu,sigma))

    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Monte Carlo Analysis for stock')
    plt.show()

montePlotter()

def simulation():
    runs = 1000
    simulations = np.zeros(runs)

    for run in range(runs):
        simulations[run] = stock_monte_carlo(last_price,days,mu,sigma)[days-1]
        
    q = np.percentile(simulations,1)

    plt.hist(simulations,bins=200)

    plt.figtext(0.6,0.8,s='Start price: $%.2f' % last_price)
    plt.figtext(0.6,0.7,'Mean final price: $%.2f' % simulations.mean())
    plt.figtext(0.6,0.6,'VaR(0.99): $%.2f' % (last_price - q,))
    plt.figtext(0.15,0.6,'q(0.99): $%.2f' % q)
    plt.axvline(x=q,linewidth=4,color='red')
    plt.title(u'Final price distribution for stock after %s days' % days,weight='bold')
    plt.show()

simulation()
