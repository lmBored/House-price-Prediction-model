import random
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import hvplot.pandas 
from sklearn import linear_model 
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from math import log
sns.set_style('whitegrid') 
import warnings
warnings.filterwarnings('ignore') 
from scipy import stats, linalg
from matplotlib import rcParams
import scipy.stats as st
import folium 
from folium import plugins

%matplotlib inline

data = pd.read_csv('https://raw.githubusercontent.com/lmBored/HousePrice/main/house_data.csv', parse_dates = ['date']) 
data.head(2)
house_df = data
X = house_df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'view', 'grade', 'sqft_above', 'sqft_basement', ]]

y = house_df['price']
house_df.shape
house_df.head()
house_df.info()
house_df.describe()
house_df.plot.scatter('sqft_living', 'price')
house_df.plot.scatter('grade', 'price')
house_df.plot.scatter('view', 'price')
house_df.plot.scatter('floors', 'price')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)

print(lin_reg.intercept_)

coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df

pred = lin_reg.predict(X_test)
pd.DataFrame({'True Values': y_test, 'Predicted Values': pred}).hvplot.scatter(x='True Values', y='Predicted Values')
pd.DataFrame({'Error Values': (y_test - pred)}).hvplot.kde()
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])