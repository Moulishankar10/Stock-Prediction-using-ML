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

model = PolynomialFeatures(degree = 4)
model_x = model.fit_transform(x)
mod = LinearRegression()
mod.fit(model_x,y)

res = mod.predict(model.fit_transform([[x_pred]]))
print(f"\nThe Predicted Quantity of {input_product} to be sold on {input_month} -->> {round(float(res))}")


#To visualise the accuracy

"""
plt.plot(data["Month"],y,color = "red")
plt.plot(data["Month"],mod.predict(model.fit_transform(x)),color = "blue")
plt.show()
"""

