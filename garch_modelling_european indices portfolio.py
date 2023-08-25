#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
# Load the data
prices = pd.read_csv('europe indices monthly.csv', index_col='Date', parse_dates=True)

# Calculate the returns and create a new DataFrame
returns = prices.pct_change().dropna()

# Set the weights
weights = np.full(len(prices.columns), 1/len(prices.columns))

# Set the initial investment and monthly investment
initial_investment = 100
monthly_investment = 100
target_investment=14400
# Initialize the portfolio value with the initial investment
portfolio_value = [initial_investment]



# Iterate over the months and calculate the portfolio value
for i in range(len(prices) - 1):
    # Rebalance the portfolio every month
    if (i + 1) % 12 == 0:
        weights = np.full(len(prices.columns), 1/len(prices.columns))
    # Calculate the returns for the current month
    returns_current_month = returns.iloc[i]
    # Calculate the portfolio return for the current month
    portfolio_return = np.dot(weights, returns_current_month)
    # Calculate the portfolio value for the current month
    portfolio_value_month = portfolio_value[i] * (1 + portfolio_return)
    # Add the monthly investment to the portfolio value
    portfolio_value_month += monthly_investment
    # Add the portfolio value to the list of portfolio values
    portfolio_value.append(portfolio_value_month)

# Convert the list of portfolio values to a numpy array
portfolio_value = np.array(portfolio_value)

# Add the portfolio value as a new column in the prices DataFrame
prices['Portfolio Value'] = portfolio_value




# Calculate the total return, volatility, and Sharpe ratio
total_return = (prices['Portfolio Value'].iloc[-1] / target_investment - 1) 




cov_matrix = np.cov(returns.T)
portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(12) * 100
sharpe_ratio = ((total_return) - 0.0363) / returns.std()
#sharpe_ratio = ((total_return) - 0.0363) / (portfolio_volatility) * 0.01
print("Total return: {:.2f}".format(total_return))
print("Volatility: {:.2f}".format(portfolio_volatility))
print("Sharpe ratio: {:.2f}".format(sharpe_ratio.iloc[0]))
print(prices["Portfolio Value"])


# In[2]:






monthly_returns = ((portfolio_value[1:] - 100) / portfolio_value[:-1] )- 1
print(monthly_returns)
plt.figure(figsize=(12,8))
plt.plot(monthly_returns)
plt.xlabel('Month')
plt.ylabel('Monthly Return')
plt.title('Monthly Returns of the Portfolio')
plt.show()


# In[3]:


import datetime as dt
import sys
import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import pandas as pd
from pandas_datareader import data as web
import seaborn as sns
from pylab import rcParams 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from arch import arch_model
from numpy.linalg import LinAlgError
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot, moment
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
from sklearn.model_selection import TimeSeriesSplit
import warnings


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)
rcParams['figure.figsize'] = 8,4


# In[5]:


df = pd.DataFrame(monthly_returns, columns=['Returns'])

# Now 'df' is a DataFrame containing your array
print(df)


# In[6]:


stdv=df['Returns'].std()
print(stdv)


# In[7]:


# Specify GARCH model assumptions
basic_gm = arch_model(df['Returns'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
# Fit the model
gm_result = basic_gm.fit()


# In[8]:


# Display model fitting summary
print(gm_result.summary())


# In[9]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
gm_vol = gm_result.conditional_volatility
def evaluate(observation, forecast): 
    # Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print(f'Mean Absolute Error (MAE): {round(mae,3)}')
    # Call sklearn function to calculate MSE
    mse = mean_squared_error(observation, forecast)
    print(f'Mean Squared Error (MSE): {round(mse,3)}')
    return mae, mse

# Backtest model with MAE, MSE
evaluate(df['Returns'].sub(df['Returns'].mean()).pow(2), gm_vol**2)


# In[112]:


gm_vol = gm_result.conditional_volatility

# Plot the actual rets volatility
plt.plot(df['Returns'].sub(df['Returns'].mean()).pow(2), 
         color = 'grey', alpha = 0.52, label = 'Daily Volatility')

# Plot EGARCH  estimated volatility
plt.plot(gm_vol**2, color = 'red', label = 'GARCH Volatility')

plt.legend(loc = 'upper right')
plt.show()


# In[11]:


# Plot fitted results
gm_result.plot()
plt.show()


# In[12]:


# Make 5-period ahead forecast
gm_forecast = gm_result.forecast(horizon = 5)

# Print the forecast variance
print(gm_forecast.variance[-1:])


# In[13]:


# Obtain model estimated residuals and volatility
gm_resid = gm_result.resid
gm_std = gm_result.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std

# Plot the histogram of the standardized residuals
plt.figure(figsize=(7,4))
sns.distplot(gm_std_resid, norm_hist=True, fit=stats.norm, bins=10, color='r')
plt.legend(('normal', 'standardized residuals'))
plt.show()


# In[14]:


# Specify GARCH model assumptions
skewt_gm = arch_model(df['Returns'], p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'skewt')

# Fit the model
skewt_result = skewt_gm.fit(disp = 'off')

# Get model estimated volatility
skewt_vol = skewt_result.conditional_volatility


# In[113]:


# Plot model fitting results
plt.plot(skewt_vol, color = 'red', label = 'Skewed-t Volatility')
plt.plot(df['Returns'], color = 'grey', 
         label = 'Daily Returns', alpha = 0.52)
plt.legend(loc = 'upper right')
plt.show()


# In[118]:


#Specify GJR-GARCH model assumptions
gjr_gm = arch_model(df['Returns'], p = 1, q = 1, o = 1, vol = 'EGARCH', dist = 't')

# Fit the model
gjrgm_result = gjr_gm.fit(disp = 'off')

# Print model fitting summary
print(gjrgm_result.summary())


# In[120]:


# Specify EGARCH model assumptions
egarch_gm = arch_model(df['Returns'], p = 1, q = 1, o = 1, vol = 'EGARCH', dist = 't')

# Fit the model
egarch_result = egarch_gm.fit(disp = 'off')

# Print model fitting summary
print(egarch_result.summary())


# In[114]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
def evaluate1(observation, forecast): 
    # Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print(f'Mean Absolute Error (MAE): {round(mae,3)}')
    # Call sklearn function to calculate MSE
    mse = mean_squared_error(observation, forecast)
    print(f'Mean Squared Error (MSE): {round(mse,3)}')
    return mae, mse

# Backtest model with MAE, MSE
evaluate1(df['Returns'].sub(df['Returns'].mean()).pow(2), gjrgm_vol**2)


# In[20]:


gjrgm_vol = gjrgm_result.conditional_volatility

# Plot the actual Bitcoin volatility
plt.plot(df['Returns'].sub(df['Returns'].mean()).pow(2), 
         color = 'grey', alpha = 0.4, label = 'Daily Volatility')

# Plot EGARCH  estimated volatility
plt.plot(gjrgm_vol**2, color = 'red', label = 'GJRGARCH Volatility')

plt.legend(loc = 'upper right')
plt.show()


# In[121]:



egarch_vol = egarch_result.conditional_volatility

# Plot the actual Bitcoin returns
plt.plot(df['Returns'], color = 'grey', alpha = 0.4, label = 'Price Returns')

# Plot GJR-GARCH estimated volatility
plt.plot(gjrgm_vol, color = 'blue', label = 'GJR-GARCH Volatility')

# Plot EGARCH  estimated volatility
plt.plot(egarch_vol, color = 'red', label = 'EGARCH Volatility')

plt.legend(loc = 'upper right')
plt.show()


# In[122]:


# Print each models BIC
print(f'GJR-GARCH BIC: {gjrgm_result.bic}')
print(f'\nEGARCH BIC: {egarch_result.bic}')


# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.model_selection import train_test_split

# Assuming you have already defined df and fitted the GARCH model (gm_result)

# Split the data into training and testing sets (80-20 split)
train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]
print(test_data)
# Fit the GARCH model on the training data
gm_result = basic_gm.fit()

# Predict volatility on both training and testing data
train_volatility = np.sqrt(gm_result.conditional_volatility.iloc[:train_size])
test_volatility = np.sqrt(gm_result.conditional_volatility.iloc[train_size:])

# Plot actual vs. predicted volatility
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['Returns'], color='grey', label='Train Returns', alpha=0.6)
plt.plot(test_data.index, test_data['Returns'], color='blue', label='Test Returns', alpha=0.6)
plt.plot(train_volatility.index, train_volatility, color='red', label='Predicted Train Volatility')
plt.plot(test_volatility.index, test_volatility, color='green', label='Predicted Test Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend(loc='upper left')
plt.title('Actual vs. Predicted Volatility')
plt.show()


# In[123]:


last_obs = 113  # Index of the last return you want to include

# Filter the data to include observations up to the last observation index
#filtered_data = df.iloc[:last_observation_index]

# Fit the ARCH/GARCH model to the filtered data
am = arch_model(df['Returns'], p=1, q=1, vol='GARCH', dist='normal')
res = am.fit(disp='off',last_obs=last_obs)
print(res)
# Now 'res' contains the model results, and the last observation is the 128th return

# Obtain model estimated residuals and volatility
gm_resid = res.resid
gm_std = res.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std

# Obtain the empirical quantiles
q = gm_std_resid.quantile([.01, .05])
print(q)
#print(params)
print(res)


# In[137]:


value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * -q[None, :]
value_at_risk = pd.DataFrame(value_at_risk, columns=['1%', '5%'], index=cond_var.index)
value_at_risk.describe()


# In[147]:


ax = value_at_risk.plot(legend=False, figsize=(12,6))
xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])

 #Assuming you want to start at index 114
start_index = 113
rets_2020 = df['Returns'].iloc[-30]
#rets_2020.name = 'Portfolio Return'


c = []
for idx in value_at_risk.index:
    if rets_2020[idx] > -value_at_risk.loc[idx, '5%']:
        c.append('#000000')
    elif rets_2020[idx] < -value_at_risk.loc[idx, '1%']:
        c.append('#BB0000')
    else:
        c.append('#BB00BB')
        
c = np.array(c, dtype='object')
for color in np.unique(c):
    sel = c == color
    ax.scatter(
        rets_2020.index[sel],
        -rets_2020.loc[sel],
        marker=markers[color],
        c=c[sel],
        label=labels[color])
    
ax.set_title('Filtered Historical Simulation VaR')
ax.legend(frameon=False, ncol=3)

plt.show()


# In[139]:


import pandas as pd
from arch import arch_model

# Assuming you have a DataFrame named df with a column 'Returns'

# Calculate the number of rows in df
n_rows = len(df)

# Set last_obs to 113
last_obs = 113

# Create the ARCH model
am = arch_model(df['Returns'], p=1, q=1, vol='GARCH', dist='t')
res = am.fit(disp='off', last_obs=last_obs)

# Set the forecast start index to 114
forecast_start_index = 114

# Forecast
forecasts = res.forecast(start=forecast_start_index)

# Get conditional mean and variance from index 114 onwards
cond_mean = forecasts.mean.iloc[forecast_start_index:]
cond_var = forecasts.variance.iloc[forecast_start_index:]

# Calculate quantiles
q1 = am.distribution.ppf([0.01, 0.05], res.params[4])
print(q)


# In[140]:


value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * -q1[None, :]
value_at_risk = pd.DataFrame(value_at_risk, columns=['1%', '5%'], index=cond_var.index)
value_at_risk.describe()
print(value_at_risk)


# In[148]:


ax = value_at_risk.plot(legend=False, figsize=(12,6))
xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])

start_index = 114
rets_2020 = df['Returns'].iloc[start_index:]
rets_2020.name = 'Portfolio Return'

c = []
for idx in value_at_risk.index:
    if rets_2020[idx] < -value_at_risk.loc[idx, '5%']:
        c.append('#000000')
    elif rets_2020[idx] < -value_at_risk.loc[idx, '1%']:
        c.append('#BB0000')
    else:
        c.append('#BB00BB')
        
c = np.array(c, dtype='object')

labels = {
    
    '#BB0000': '5% Exceedence',
    '#BB00BB': '1% Exceedence',
    '#000000': 'No Exceedence'
}

markers = {'#BB0000': 'x', '#BB00BB': 's', '#000000': 'o'}

for color in np.unique(c):
    sel = c == color
    ax.scatter(
        rets_2020.index[sel],
        -rets_2020.loc[sel],
        marker=markers[color],
        c=c[sel],
        label=labels[color])
    
ax.set_title('Parametric VaR')
ax.legend(frameon=False, ncol=3)

plt.show()


# In[ ]:




