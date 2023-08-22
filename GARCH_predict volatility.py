#!/usr/bin/env python
# coding: utf-8

# In[2]:


from arch import arch_model
from arch.__future__ import reindexing
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('europedailyprices.csv', parse_dates=True, index_col='Date')


# In[4]:


returns = df.pct_change().dropna()
print(returns)


# In[5]:


daily_volatility = returns['EUSTX50'].std()
print('Daily volatility: ', '{:.2f}%'.format(daily_volatility))

monthly_volatility = math.sqrt(21) * daily_volatility
print ('Monthly volatility: ', '{:.2f}%'.format(monthly_volatility))

annual_volatility = math.sqrt(252) * daily_volatility
print ('Annual volatility: ', '{:.2f}%'.format(annual_volatility ))


# In[7]:


scaled_returns = returns * 100
garch_model = arch_model(returns['EUSTX50'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')

gm_result = garch_model.fit(disp='off')
print(gm_result.params)

print('\n')

gm_forecast = gm_result.forecast(horizon = 5)
print(gm_forecast.variance[-1:])


# In[9]:


scaled_returns = returns * 100
rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns['EUSTX50'][:-(test_size-i)]
    model = arch_model(train, p=1, q=1)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
rolling_predictions = pd.Series(rolling_predictions, index=returns['EUSTX50'].index[-365:])

plt.figure(figsize=(10,4))
plt.plot(rolling_predictions)
plt.title('Rolling Prediction')
plt.show()


# In[11]:


plt.figure(figsize=(12,4))
plt.plot(returns['EUSTX50'][-365:])
plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast EUSTX50')
plt.legend(['True Daily Returns', 'Predicted Volatility'])
plt.show()


# In[ ]:




