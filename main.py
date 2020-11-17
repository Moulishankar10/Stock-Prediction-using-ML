# STOCK PREDICTION USING ML

# DEVELOPED BY: 
# MOULISHANKAR M R
# VIGNESHWAR RAVICHANDAR

# IMPORTING REQUIRED MODULES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.ensemble import RandomForestRegressor


def mlmodel():
    # IMPORTING DATA
    data = pd.read_csv("data/Quantity Sold.csv")

    # INPUT DATA
    print("\nEnter the following details as what you want to predict!")
    input_month = input("\nEnter the time period (MM-YYYY) : ")
    input_product = input("\nEnter the product : ").upper()

    # PREPROCESSING DATA
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

    # FITTING MODEL
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(x, y)

    # PREDICTING MODEL
    res = regressor.predict([[x_pred]])
    
    # DISPLAYING RESULTS
    print(f"\nThe Predicted Quantity of {input_product} to be sold on {input_month} -->> {round(float(res))}")
    print("\nAccuracy : ",regressor.score(x,y))

    #TO VISUALISE THE ACCURACY

    x_grid = np.arange(min(x), max(x), 0.01)
    x_grid = x_grid.reshape((len(x_grid), 1))
    plt.plot(x, y, color = 'red')
    plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
    plt.show()
    

print(
    '''

                     WELCOME TO STOCK PREDICTION PROGRAM !

            FOR BEING COMFORTABLE WITH THIS PROGRAM, PLEASE PROVIDE 
                      
                           THE FOLLOWING DETAILS


''')

print('''\n 
                        WHAT IS YOUR OPERATING SYSTEM ?

                              1. WINDOWS

                              2. LINUX / MAC OS

''')

opt = int(input("\nYOUR OPTION : "))

if opt == 1:
     print("\nNOW YOU WILL BE REDIRECTED TO 'requirements.bat'")
     os.system('./requirements.bat')
     mlmodel()

if opt == 2:
    print("\nNOW YOU WILL BE REDIRECTED TO 'requirements.sh'")
    os.system('bash requirements.sh')
    mlmodel()