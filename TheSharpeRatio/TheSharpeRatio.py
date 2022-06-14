# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import io
st.title('Risk and Returns: The Sharpe Ratio')
st.header('Mini-Project from DataCamp')
st.write("""
     In 1966, Professor William Sharpe introduced to reward-to-variability ratio. It looks at the difference between 2 investments and compares the average difference to the standard deviation. A higher Sharpe ratio means that the reward will be higher for a given amount of risk. This script calculates a simplified version of the Sharpe ratio for Facebook and Amazon, using the S&P 500 as a benchmark. 
""")
# Settings to produce nice plots in a Jupyter notebook
plt.style.use('fivethirtyeight')
st.header('Data Processing')
st.write("We will read in our data from csv file format and set the index to a datetime object.")
# Reading in the data
st.code('''
stock_data = pd.read_csv('dataset/stock_data.csv',parse_dates=True,index_col='Date')
benchmark_data = pd.read_csv('dataset/benchmark_data.csv',parse_dates=True,index_col='Date')
stock_data=stock_data.dropna()
benchmark_data=benchmark_data.dropna()
''')


stock_data = pd.read_csv('stock_data.csv',parse_dates=True,index_col='Date')

benchmark_data = pd.read_csv('benchmark_data.csv',parse_dates=True,index_col='Date')
stock_data=stock_data.dropna()
benchmark_data=benchmark_data.dropna()
st.write('Here, we look at the datasets')
st.caption('Stock Data')
st.dataframe(stock_data.head())
st.caption('Benchmark Data')
st.dataframe(benchmark_data.head())
"""
print('Stocks\n')
print(stock_data.info())
print('\nBenchmarks\n')
print(benchmark_data.info())
print(stock_data.head())
"""
st.header('Visualize the Stock Data')
st.line_chart(stock_data)
#stock_data.plot(subplots=True,title='Stock Data')
st.header('Visualize the Benchmark Data')
st.line_chart(benchmark_data['S&P 500'])
#benchmark_data.plot(title='S&P 500')
#print(benchmark_data.describe())
stock_returns = stock_data.pct_change()
stock_returns.plot()
stock_returns.describe()
st.header('Calculations')
st.subheader('S&P 500 Daily Return')
st.code('''
sp_returns = benchmark_data['S&P 500'].pct_change()
''')
st.caption('First 5 rows')
st.write(benchmark_data[['S&P 500']].pct_change().head())
st.write('Visualization')
sp_returns = benchmark_data['S&P 500'].pct_change()
st.line_chart(sp_returns)
#sp_returns.plot()
st.subheader('Excess Returns for Amazon and Facebook')
st.code('''excess_returns =  stock_returns.sub(sp_returns,axis=0)''')
st.caption('First 5 rows')
st.write(stock_returns.sub(sp_returns,axis=0).head())
st.write('Visualization')
# calculate the difference in daily returns
excess_returns =  stock_returns.sub(sp_returns,axis=0)
st.line_chart(excess_returns)
#excess_returns.plot()
#excess_returns.describe()
st.subheader('Calcuate the Mean of Excess Returns')
st.code('''avg_excess_return = excess_returns.mean()''')
avg_excess_return = excess_returns.mean()
st.bar_chart(avg_excess_return)
#avg_excess_return.plot.bar(title='Mean of the Return Difference')
st.subheader('Calculate Standard Deviation of Excess Returns')
st.code('''sd_excess_return = excess_returns.std()''')
sd_excess_return = excess_returns.std()
st.bar_chart(sd_excess_return)
#sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')
st.header('Putting it Together')
st.write('Now, find the ratio of the mean excess reutrns and the standard deviation of excess returns. This is the Sharpe Ratio and indicates how much more or less the investment opportunity under consideration yields per unit of risk. The Sharpe Ratio is often annualized by multiplying it by the square root of the number of periods. Daily data was used as input, so the square root of the number of trading days: √252 ')
st.code('''daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio*annual_factor''')
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio*annual_factor
st.header('Our Result')
st.write('Annualized Sharpe Ratio: Stocks vs S&P 500')
st.line_chart(annual_sharpe_ratio)
st.write('Amazon:',annual_sharpe_ratio[0])
st.write('Facebook:',annual_sharpe_ratio[1])
st.header('Conclusion')
st.write("Amazon's Sharpe Ratio was twice as high as Facebook's in 2016. This means that an investment in Amazon returned twice as much compared to the S&P 500 for each unit of risk an investor would have assumed. Therefore, the investment in Amazon would have been more attractive. ")
#annual_sharpe_ratio.plot(title='Annualized Sharpe Ratio: Stocks vs S&P 500')


#%%
