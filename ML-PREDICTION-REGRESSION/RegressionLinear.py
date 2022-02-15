import pandas as pd
import numpy as np

from sklearn import linear_model

df = pd.read_csv('dataset.csv')
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['Cycle_Index', 'Current', 'Voltage', 'Discharge_Capacity']],df.Charge_Capacity)

print("________________________________________________________________")

print("The coefficients of the dependent variables - ...")
print(reg.coef_)

print("________________________________________________________________")

print("The intercept of the equation : ")
print(reg.intercept_)
print("________________________________________________________________")
print("Please enter the details : ")
a = input("Cucle Index        : ")
b = input("Current (A)        : ")
c = input("Voltage (V)        : ")
d = input("Discharge Capacity : ")

CCPrediction = reg.predict( [[a,b,c,d]] )

print("****Charge Capacity Predicted value*****")
print(CCPrediction)
