#!/usr/bin/env python
# coding: utf-8

# # THE SPARK FOUNDATION INTERNSHIP
# # #Task 1: To predict the percentage of student based on the number of study hours.
# #### Author: M. Harish Babu

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing the dataset
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Dataset successfully imported")


# In[8]:


#displaying the first 10 rows of the dataset
data.head(10)


# In[9]:


#getting the summary of the dataset
data.describe()


# # Visualising the Data

# In[13]:


#plotting score distribution 
data.plot(x="Hours",y="Scores",style=".")
plt.title("Hours vs Percentage")
plt.xlabel("Hours Spent")
plt.ylabel("Percentage Scored")
plt.show()


# # Preparing the data

# In[14]:


#preparing X and Y as attributes and labels respectively
X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values


# # Splitting datasets

# In[15]:


#splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[16]:


#linear regression using scikit-learn
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print("Training successfully completed")


# In[17]:


print("Coefficient is :", regressor.coef_)
print("Intercept is :", regressor.intercept_)


# In[18]:


#Plotting regression line
line = regressor.coef_*X + regressor.intercept_


# In[20]:


#plotting for test data
plt.scatter(X, Y)
plt.plot(X, line, color = "red", label = "Regression Line")
plt.legend()
plt.show()


# # Predictions

# In[21]:


print(X_test)
Y_pred = regressor.predict(X_test)


# In[22]:


#Comparing actual vs predicted values
df=pd.DataFrame({'Actual values' : Y_test, 'Predicted values' : Y_pred})
df


# In[23]:


#Estimation of train and test scores
print("Training score is :",regressor.score(X_train, Y_train))
print("Test score is :",regressor.score(X_test, Y_test))


# In[27]:


#plotting actual vs predicted
df.plot(kind = 'bar', figsize = (11,5))
plt.grid(which = 'major', linewidth = '0.4', color = 'yellow')
plt.grid(which = 'major', linewidth = '0.4', color = 'red')
plt.show()


# # Predicting with new data

# In[29]:


hours = 9.25
test = np.array([hours])
test = test.reshape(-1,1)
own_pred = regressor.predict(test)
print("Number of hours taken is : {}".format(hours))
print("Predicted score is : {}".format(own_pred[0]))


# # Model Evaluation 

# In[30]:


print("Mean Absolute Error is :", metrics.mean_absolute_error(Y_test,Y_pred))
print("Mean Squared Error is :", metrics.mean_squared_error(Y_test,Y_pred))
print("Root Mean Squared Error is :", np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
print("R-2 is :", metrics.r2_score(Y_test,Y_pred))


# In[ ]:




