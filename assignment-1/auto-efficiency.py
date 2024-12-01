import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# # Reading the data
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# data = pd.read_csv(url, sep='\s+', header=None, names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"])

# # Save the data to an Excel file with column names
# excel_file_path = 'auto_mpg_data.xlsx'
# data.to_excel(excel_file_path, index=False)

# # Print a message indicating the successful save
# print(f'Data saved to {excel_file_path}')

np.random.seed(42)

# Defining max depth and split ratio
max_depth = 6
split = 0.8    # using 80% for train and 20% for test

# Loading dataset
df = pd.read_excel('auto_mpg_data.xlsx')

# Dividing dataset into featues and output 
X = df[['cylinders','horsepower','weight']]
y = df['mpg']

sample_no = len(X)
# Calculate the index for splitting
split_index = int(split * sample_no)

# Splitting the data into training and testing sets
X_train, y_train = X[:split_index + 1], y[:split_index + 1]
X_test, y_test = X[split_index + 1:].reset_index(drop=True), y[split_index + 1:].reset_index(drop=True)

# Training the decision tree
tree = DecisionTree(criterion="information_gain", max_depth=max_depth)
tree.fit(X_train, y_train)
#tree.plot()               # Uncomment this to see the learnt decision tree

# Creating dataframe to store the rmse and mae value at different depths
results_table = pd.DataFrame(index=['rmse', 'mae'])
results_table.index.name = "Measures" 
results_table.columns.name = "Depths"


least_rmse = np.inf
least_mae = np.inf
opt_depth_rmse = 0
opt_depth_mae = 0

# calculating rmse and mae at different depths and storing it to results_table
for depth in range(1, max_depth+1):
       y_hat = tree.predict(X_test, max_depth=depth)
       r = rmse(y_hat, y_test)
       m = mae(y_hat, y_test)
       if r<least_rmse:
              least_rmse = r
              opt_depth_rmse = depth
       if m<least_mae:
              least_mae = m
              opt_depth_mae = depth

print("\n### Using Modal ###\n")
print("Least rmse = ",least_rmse, " depth = ", opt_depth_rmse)
print("Least mae = ",least_mae, " depth = ", opt_depth_mae)

# Sci-kit learn Decision Tree Regressor for using criterion='squared_error'
dt = DecisionTreeRegressor(criterion='squared_error', max_depth=max_depth, random_state=0)
dt.fit(X_train, y_train)
y_hat = pd.Series(dt.predict(X_test))

print("\n### sklearn ###")
print("\ncriterion='squared_error'")
print('rmse: ', rmse(y_hat, y_test))
print('mae: ', mae(y_hat, y_test))

# Sci-kit learn Decision Tree Regressor for using criterion='absolute_error'
dt = DecisionTreeRegressor(criterion='absolute_error', max_depth=max_depth, random_state=0)
dt.fit(X_train, y_train)
y_hat = pd.Series(dt.predict(X_test))

print("\ncriterion='absolute_error'")
print('rmse: ', rmse(y_hat, y_test))
print('mae: ', mae(y_hat, y_test))



# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn