data.drop(['id', 'date'], axis=1, inplace=True)

data['basement_present'] = data['sqft_basement'].apply(lambda x : 1 if x > 0 else 0)

data['renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

categorical_cols = ['floors', 'view', 'condition', 'grade']

for col in categorical_cols:
    dummies = pd.get_dummies(data[col], drop_first=False)
    dummies = dummies.add_prefix("{}#".format(col))
    data.drop(col, axis=1, inplace=True)
    data = data.join(dummies)

data.head()
dummies_zipcodes = pd.get_dummies(data['zipcode'], drop_first=False)
dummies_zipcodes.reset_index(inplace=True)
dummies_zipcodes = dummies_zipcodes.add_prefix("{}#".format('zipcode'))
dummies_zipcodes = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]
data.drop('zipcode', axis=1, inplace=True)
data = data.join(dummies_zipcodes)

data.dtypes
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, train_size=0.8, random_state=10)

def multiple_regression_model(train ,test, input_features):
    regr = linear_model.LinearRegression()
    regr.fit(train.as_matrix(columns=input_features), train.as_matrix(columns=['price']))
    RMSE = mean_squared_error(test.as_matrix(columns=['price']), regr.predict(test.as_matrix(columns=input_features))) ** 0.5
    
    return RMSE, regr.intercept_[0], regr.coef_

print ('RMSE: %s, intercept: %s, coefficients: %s' \ %multiple_regression_model(train_data, test_data, ['sqft_living','bathrooms','bedrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s' \ %multiple_regression_model(train_data, test_data, ['sqft_above','view#0','bathrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s' \ %multiple_regression_model(train_data, test_data, ['bathrooms','bedrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s' \ %multiple_regression_model(train_data, test_data, ['view#0','grade#12','bedrooms','sqft_basement']))
print ('RMSE: %s, intercept: %s, coefficients: %s' \ %multiple_regression_model(train_data, test_data, ['sqft_living','bathrooms','view#0']))
train_data['sqft_living_squared'] = train_data['sqft_living'].apply(lambda x: x**2)
test_data['sqft_living_squared'] = test_data['sqft_living'].apply(lambda x: x**2)
print('RMSE: %s, intercept: %s, coefficients: %s' \
      %multiple_regression_model(train_data, test_data, ['sqft_living', 'sqft_living_squared'])

train_data['sqft_living_cubed'] = train_data['sqft_living'].apply(lambda x: x**3) 
test_data['sqft_living_cubed'] = test_data['sqft_living'].apply(lambda x: x**3)

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2) 
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)

train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))
train_data_2, validation_data = train_test_split(train_data, train_size = 0.75, random_state = 50)

print(data.shape)
print(train_data_2.shape)
print(validation_data.shape)
print(test_data.shape)

def RMSE(train, validation, features, new_input):
    features_list = list(features)
    features_list.append(new_input)
    regr = linear_model.LinearRegression() 
    regr.fit(train.as_matrix(columns = features_list), train.as_matrix(columns = ['price'])) 
    RMSE_train = mean_squared_error(train.as_matrix(columns = ['price']), 
                              regr.predict(train.as_matrix(columns = features_list)))**0.5 
    RMSE_validation = mean_squared_error(validation.as_matrix(columns = ['price']), 
                              regr.predict(validation.as_matrix(columns = features_list)))**0.5 
    return RMSE_train, RMSE_validation 

input_list = train_data_2.columns.values.tolist() # list of column name
input_list.remove('price')
regression_greedy_algorithm = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])  
i = 0
temp_list = []

while i < len(train_data_2.columns)-1:
    temp = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])
    for p in input_list:
        RMSE_train, RMSE_validation = RMSE(train_data_2, validation_data, temp_list, p)
        temp = temp.append({'feature':p, 'train_error':RMSE_train, 'validation_error':RMSE_validation}, ignore_index=True)
        
    temp = temp.sort_values('train_error') 
    best = temp.iloc[0,0]
    temp_list.append(best)
    regression_greedy_algorithm = regression_greedy_algorithm.append({'feature': best, 'train_error': temp.iloc[0,1], 'validation_error': temp.iloc[0,2]}, ignore_index=True) 
    input_list.remove(best) 
    i += 1

regression_greedy_algorithm['index'] = regression_greedy_algorithm.index

regression_greedy_algorithm
plt.figure(figsize=(8,8))
sns.lineplot(data=regression_greedy_algorithm.loc[:, ['train_error', 'validation_error']])

plt.show()
greedy_algo_features_list = regression_greedy_algorithm['feature'].tolist()[:] 
test_error, _, _ = multiple_regression_model(train_data_2, test_data, greedy_algo_features_list)
print ('test error (RMSE) is: %s' %test_error)
test_temp = []
for cnt in range(regression_greedy_algorithm.shape[0]):
    greedy_algo_features_list = regression_greedy_algorithm['feature'].tolist()[:cnt+1] 
    test_error, _, _ = multiple_regression_model(train_data_2, test_data, greedy_algo_features_list)
    test_temp.append(test_error)