import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("dataset.csv")
headCheck = df.head()
print(headCheck)

# Get dataset inforrmation
infoData = df.info()
print(infoData)

# Get mathematical description
mathdes = df.describe()
print(mathdes)

'''# HISTO PLOT
plt.figure(figsize=(8,4))
sns.histplot(df['Charge_Capacity'])
#plt.show()'''

'''# Scatter plot
plt.figure(figsize=(12,8))
sns.scatterplot(y='Charge_Capacity',x='Cycle_Index',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(y='Discharge_Capacity',x='Cycle_Index',data=df)
plt.show()
'''

'''plt.figure(figsize=(12,8))
sns.scatterplot(y='Discharge_Capacity',x='Charge_Capacity',data=df)
plt.show()'''

'''# ---- HEATMAP DIAGRAM TO SHOW DIFFERENT CORRELATION VARIABLE VALUES
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()
'''

# Dataset Split into TRAINING AND TESTING
X = df.drop('Charge_Capacity',axis=1)
y = df['Charge_Capacity']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# ----- DATASET SCALING

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("####################################################")
print(X_test)
print("####################################################")
# model ARCHITECTURE ...
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Dense(8,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# TRAINING CODE ...

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', 'val_acc', mode='min', verbose=1, patience=10)
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=5, callbacks=[early_stop])


model.save('regressionTwo.h5')
print("Model saved ...")


losses = pd.DataFrame(model.history.history)
lossPlot = losses.plot()
plt.show()

accuracy = pd.DataFrame(model.history.history)
accPlot = accuracy.plot()
plt.show()

# MODEL EVALUATION

from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = model.predict(X_test[5.03597122e-02,1.42836738e-04,4.83815029e-01,2.73626749e-01])
mean_absolute_error(y_test,predictions)
predictionFinal = np.sqrt(mean_squared_error(y_test,predictions))
print(predictionFinal)
