# STOCK PREDICTION USING ML

# DEVELOPED BY: 
# MOULISHANKAR M R
# VIGNESHWAR RAVICHANDAR

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# IMPORTING DATA
data = pd.read_csv("data/Quantity Sold.csv")

# INPUT
print("\nEnter the following details as what you want to predict!")
input_month = input("\nEnter the time period (MM-YYYY) : ")
input_product = input("\nEnter the product : ").upper()

x = []
initial_str = data["Month"][0]
initial = dt.datetime(int(initial_str[-4:]),int(initial_str[:2]),1)

x_str = dt.datetime(int(input_month[-4:]),int(input_month[:2]),1)
x_pred = (x_str.year - initial.year) * 12 + (x_str.month - initial.month)

for i in range(len(data["Month"])):
    final_str = data["Month"][i]
    final = dt.datetime(int(final_str[-4:]),int(final_str[:2]),1)
    diff = (final.year - initial.year) * 12 + (final.month - initial.month)
    x.append(diff)

x = np.array(x,dtype=int)
x = x.reshape(len(x),1)

y = data[input_product].values
y = np.array(y,dtype=int)
y = y.reshape(len(y),1)

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

res = sc_y.inverse_transform(regressor.predict(sc_x.transform([[x_pred]])))


print(f"\nThe Predicted Quantity of {input_product} to be sold on {input_month} -->> {round(float(res))}")


#To visualise the accuracy

"""
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.scatter(x_pred, res, color='green')
plt.show()

"""

