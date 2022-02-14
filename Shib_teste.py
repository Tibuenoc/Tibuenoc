#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd      
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns 
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("/Users/TiagoBueno/Desktop/SHIB-USD.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.isnull().values.any()


# In[7]:


df.corr()


# In[8]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)


# In[9]:


df['Close'].duplicated().any()


# In[10]:


fig = px.line(df, x = "Date", y = ["Open", "High", "Low", "Close"], title = "Interactive Time Series Analysis")
fig.show()


# In[11]:


x = df[[ 'Open', 'High', 'Low','Volume']]
y = df['Close']


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[13]:


from sklearn.preprocessing import StandardScaler
#Initalise standard scaler
scaler = StandardScaler()
#Fit the scaler using X_train data
scaler.fit(x_train)
#Transform x_train and X_test using the scaler and convert back to DataFrame
X_train = pd.DataFrame(scaler.transform(x_train), columns = x_train.columns)
X_test = pd.DataFrame(scaler.transform(x_test), columns = x_test.columns)


# In[21]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

#Use Model to Make Predictions
y_pred = model.predict(x_test)

#Get Intercept & Coefficients
print(model.intercept_)
coef = pd.DataFrame(model.coef_, x_train.columns, columns=['Coef'])

#Get MSE & MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('MSE:',mean_squared_error(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# In[27]:


#Compare Original Price x Predict Price 
plt.scatter(df.Close, model.predict(x))
plt.xlabel("Original Price (Close)")
plt.ylabel("Predict Price")
plt.title("Original Price x Predict Price")
plt.show()


# In[18]:


import pandas_profiling
from pandas_profiling import ProfileReport

profile = ProfileReport(df[[ 'Open', 'High', 'Low','Volume', 'Close']], title = "Pandas Profiling Report", explorative = True)

profile.to_widgets()


# In[ ]:




